#!/usr/bin/env python3
"""
Pretrain the ALD lipidomics foundation model (V3).

Single-GPU:
    python pretrain_transformer.py --config configs/pretrain.yaml

Multi-GPU (4 GPUs on one node):
    torchrun --nproc_per_node=4 pretrain_transformer.py --config configs/pretrain.yaml

Multi-node (2 nodes × 4 GPUs via SLURM / torchrun):
    See slurm_pretrain.sh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP

from data.loaders import make_loader
from data.pretrain_dataset import PretrainCSVIterableDataset, discover_family_csvs
from data.preprocessing import compute_norm_stats_from_files
from data.schema import build_group_ids, infer_feature_schema, save_schema
from model.heads import SharedRegressionHead
from model.masking import MaskingConfig
from model.transformer_encoder import TabularTransformerEncoder
from model.vae import VAEBottleneck
from training.checkpointing import save_checkpoint
from training.optimizer import build_optimizer, get_cosine_schedule_with_warmup
from training.trainer_pretrain import PretrainTrainer, PretrainTrainerConfig
from utils.config import load_yaml_config
from utils.ddp import cleanup, init_distributed, is_main
from utils.io import ensure_dir, save_json
from utils.logging_utils import make_logger
from utils.seed import set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--synthetic_root', type=str, default=None)
    ap.add_argument('--input_folders', nargs='+', default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help='Validate startup with 2 files × 2 steps then exit. '
                         'Runs as single rank; does not require torchrun.')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    dry_run: bool = args.dry_run

    if args.synthetic_root is not None:
        cfg['data']['synthetic_root'] = args.synthetic_root
    if args.input_folders is not None:
        cfg['data']['input_folders'] = args.input_folders

    rank, local_rank, world_size = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    set_seed(cfg['experiment']['seed'] + rank)

    run_name = cfg['experiment']['name'] + '_' + '-'.join(cfg['data']['input_folders'])
    run_dir = ensure_dir(Path(cfg['paths']['output_root']) / 'runs' / 'pretrain' / run_name)
    logger = make_logger(run_dir / 'console.log') if is_main(rank) else make_logger(run_dir / f'console_rank{rank}.log')

    # ------------------------------------------------------------------ data
    csv_files = discover_family_csvs(
        cfg['data']['synthetic_root'],
        cfg['data']['input_folders'],
        cfg['data'].get('file_glob', '*.csv'),
        cfg['data'].get('max_files'),
    )
    if not csv_files:
        raise RuntimeError('No synthetic CSV files found.')

    if dry_run:
        csv_files = csv_files[:2]

    if is_main(rank):
        prefix = '[DRY-RUN] ' if dry_run else ''
        logger.info(f'{prefix}Found {len(csv_files)} CSV files. World size: {world_size}')

    df0 = pd.read_csv(csv_files[0], nrows=32)
    schema = infer_feature_schema(
        df0,
        sex_col=cfg['data']['sex_col'],
        target_column=cfg['data']['target_column'],
        id_col=cfg['data']['id_col'],
        exclude_cols=cfg['data'].get('exclude_cols', []),
    )
    if is_main(rank):
        save_schema(schema, run_dir / 'feature_schema.json')
        logger.info(f'Feature count: {len(schema.feature_cols)}')

    # Norm stats: computed on rank 0 only then broadcast to save time.
    # Cached to disk so re-runs skip the expensive full-dataset pass.
    norm_stats_cache = run_dir / 'norm_stats.npz'
    if is_main(rank):
        if not dry_run and norm_stats_cache.exists():
            _c = np.load(norm_stats_cache)
            from data.preprocessing import NormStats
            norm_stats = NormStats(mean=_c['mean'], std=_c['std'])
        else:
            norm_stats = compute_norm_stats_from_files(
                csv_files,
                schema.feature_cols,
                chunksize=cfg['data'].get('chunksize', 4096),
                max_rows_per_file=cfg['data'].get('max_rows_per_file'),
            )
            if not dry_run:
                np.savez(norm_stats_cache, mean=norm_stats.mean, std=norm_stats.std)
        mean_t = torch.from_numpy(norm_stats.mean)
        std_t = torch.from_numpy(norm_stats.std)
    else:
        mean_t = torch.zeros(len(schema.feature_cols))
        std_t = torch.ones(len(schema.feature_cols))

    if world_size > 1:
        import torch.distributed as dist
        mean_t = mean_t.to(device)
        std_t = std_t.to(device)
        dist.broadcast(mean_t, src=0)
        dist.broadcast(std_t, src=0)
        from data.preprocessing import NormStats
        norm_stats = NormStats(
            mean=mean_t.cpu().numpy().astype('float32'),
            std=std_t.cpu().numpy().astype('float32'),
        )

    train_files, val_files = train_test_split(
        csv_files,
        test_size=cfg['validation']['split_ratio'],
        random_state=cfg['experiment']['seed'],
    )

    chunksize = cfg['data'].get('chunksize', 4096)
    max_rows_per_file = cfg['data'].get('max_rows_per_file')
    prefetch_factor = cfg['data'].get('prefetch_factor', 2)
    persistent_workers = cfg['data'].get('persistent_workers', True)
    num_workers = cfg['data']['num_workers']

    # infinite=True: cycle files so DDP ranks never exhaust at different steps.
    # Epoch length is controlled by steps_per_epoch (computed below).
    train_ds = PretrainCSVIterableDataset(
        train_files, schema, norm_stats,
        chunksize=chunksize, max_rows_per_file=max_rows_per_file,
        shuffle_files=True, seed=cfg['experiment']['seed'],
        rank=rank, world_size=world_size,
        infinite=True,
    )
    # Val is unsharded: all ranks evaluate the same files so step counts are
    # identical and _avg_across_ranks never deadlocks.
    val_ds = PretrainCSVIterableDataset(
        val_files, schema, norm_stats,
        chunksize=chunksize, max_rows_per_file=max_rows_per_file,
        shuffle_files=False, seed=cfg['experiment']['seed'],
        rank=0, world_size=1,
        infinite=False,
    )

    batch_size = cfg['data']['batch_size']
    # Cap workers to files per rank — spawning more workers than files means
    # some workers get empty file lists and spin uselessly with infinite=True.
    train_workers = min(num_workers, max(1, len(train_ds.csv_files)))
    val_workers = min(num_workers, max(1, len(val_ds.csv_files)))
    train_loader = make_loader(train_ds, batch_size, False, train_workers, cfg['data']['pin_memory'], prefetch_factor, persistent_workers)
    val_loader = make_loader(val_ds, batch_size, False, val_workers, cfg['data']['pin_memory'], prefetch_factor, persistent_workers)

    # ---------------------------------------------------------------- model
    group_ids = build_group_ids(schema.feature_cols)

    model_cfg = cfg['model']
    model = TabularTransformerEncoder(
        num_features=len(schema.feature_cols),
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        dim_feedforward=model_cfg.get('dim_feedforward', model_cfg['d_model'] * 2),
        dropout=model_cfg.get('dropout', 0.1),
        use_cls=model_cfg.get('use_cls', True),
        value_hidden_dim=model_cfg.get('value_hidden_dim', 64),
        value_encoder_layers=model_cfg.get('value_encoder_layers', 2),
        num_groups=7,
        group_ids=group_ids,
    ).to(device)

    mask_head = SharedRegressionHead(model_cfg['d_model']).to(device)

    vae_cfg = cfg.get('vae', {})
    vae = VAEBottleneck(
        d_model=model_cfg['d_model'],
        latent_dim=vae_cfg.get('latent_dim', 64),
        num_features=len(schema.feature_cols),
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        mask_head = DDP(mask_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        vae = DDP(vae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ------------------------------------------------------------- optimizer
    all_params = (
        list(model.parameters())
        + list(mask_head.parameters())
        + list(vae.parameters())
    )
    optimizer = build_optimizer(all_params, cfg['optimizer'])

    pretrain_cfg = cfg['pretraining']
    if dry_run:
        pretrain_cfg = dict(pretrain_cfg)
        pretrain_cfg['epochs'] = 1
        pretrain_cfg['steps_per_epoch'] = 2
        pretrain_cfg['val_steps'] = 2

    rows_per_file = cfg['data'].get('rows_per_file', 20_000)
    steps_per_epoch = pretrain_cfg.get('steps_per_epoch') or int(
        len(train_files) * rows_per_file / (batch_size * world_size)
    )
    # steps_per_epoch is derived from total train_files (same for all ranks).
    # Combined with infinite train dataset, this guarantees all ranks run the
    # same number of optimizer steps per epoch — required for DDP correctness.
    total_steps = steps_per_epoch * pretrain_cfg['epochs']
    warmup_steps = pretrain_cfg.get('warmup_steps', 2000)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main(rank):
        logger.info(f'total_steps={total_steps}  warmup={warmup_steps}  effective_batch={batch_size * world_size}')

    # --------------------------------------------------------------- trainer
    masking_cfg = MaskingConfig(
        mask_ratio=pretrain_cfg['mask_ratio'],
        mask_token_prob=pretrain_cfg['mask_token_prob'],
        random_replace_prob=pretrain_cfg['random_replace_prob'],
        keep_original_prob=pretrain_cfg['keep_original_prob'],
    )

    trainer_cfg = PretrainTrainerConfig(
        epochs=pretrain_cfg['epochs'],
        amp=pretrain_cfg['amp'] and device.type == 'cuda',
        amp_dtype=pretrain_cfg['amp_dtype'],
        grad_accum_steps=pretrain_cfg['grad_accum_steps'],
        clip_grad_norm=pretrain_cfg['clip_grad_norm'],
        run_dir=str(run_dir),
        warmup_steps=warmup_steps,
        vae_lambda_recon=pretrain_cfg.get('vae_lambda_recon', 0.5),
        vae_beta_kl=pretrain_cfg.get('vae_beta_kl', 1e-3),
        kl_beta_start=pretrain_cfg.get('kl_beta_start', 0.0),
        kl_anneal_steps=pretrain_cfg.get('kl_anneal_steps', 0),
        steps_per_epoch=steps_per_epoch,  # always a concrete int, never None
        val_steps=pretrain_cfg.get('val_steps'),
    )

    trainer = PretrainTrainer(
        model=model,
        mask_head=mask_head,
        vae=vae,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        masking_cfg=masking_cfg,
        cfg=trainer_cfg,
        feature_cols=schema.feature_cols,
        norm_stats=norm_stats,
        group_ids=group_ids,
        config_snapshot=cfg,
        logger=logger,
        rank=rank,
        world_size=world_size,
    )
    trainer.fit()
    cleanup()

    if dry_run and is_main(rank):
        logger.info('DRY-RUN PASSED — all startup paths exercised successfully.')


if __name__ == '__main__':
    main()

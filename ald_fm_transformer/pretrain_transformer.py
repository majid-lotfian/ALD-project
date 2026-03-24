#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from data.loaders import make_loader
from data.pretrain_dataset import PretrainCSVDataset, discover_family_csvs
from data.preprocessing import compute_norm_stats_from_files
from data.schema import infer_feature_schema, save_schema
from model.heads import SharedRegressionHead
from model.masking import MaskingConfig
from model.transformer_encoder import TabularTransformerEncoder
from training.optimizer import build_optimizer
from training.trainer_pretrain import PretrainTrainer, PretrainTrainerConfig
from utils.config import load_yaml_config, deep_update
from utils.io import ensure_dir
from utils.logging_utils import make_logger
from utils.seed import set_seed
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--synthetic_root', type=str, default=None)
    ap.add_argument('--input_folders', nargs='+', default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if args.synthetic_root is not None:
        cfg['data']['synthetic_root'] = args.synthetic_root
    if args.input_folders is not None:
        cfg['data']['input_folders'] = args.input_folders

    set_seed(cfg['experiment']['seed'])
    run_name = cfg['experiment']['name'] + '_' + '-'.join(cfg['data']['input_folders'])
    run_dir = ensure_dir(Path(cfg['paths']['output_root']) / 'runs' / 'pretrain' / run_name)
    logger = make_logger(run_dir / 'console.log')

    csv_files = discover_family_csvs(cfg['data']['synthetic_root'], cfg['data']['input_folders'], cfg['data'].get('file_glob', '*.csv'), cfg['data'].get('max_files'))
    if not csv_files:
        raise RuntimeError('No synthetic CSV files found for the requested families.')
    logger.info(f'Found {len(csv_files)} CSV files.')

    df0 = pd.read_csv(csv_files[0], nrows=32)
    schema = infer_feature_schema(
        df0,
        sex_col=cfg['data']['sex_col'],
        target_column=cfg['data']['target_column'],
        id_col=cfg['data']['id_col'],
        exclude_cols=cfg['data'].get('exclude_cols', []),
    )
    save_schema(schema, run_dir / 'feature_schema.json')
    norm_stats = compute_norm_stats_from_files(csv_files, schema.feature_cols)

    train_files, val_files = train_test_split(csv_files, test_size=cfg['validation']['split_ratio'], random_state=cfg['experiment']['seed'])
    train_ds = PretrainCSVDataset(train_files, schema, norm_stats)
    val_ds = PretrainCSVDataset(val_files, schema, norm_stats)
    train_loader = make_loader(train_ds, cfg['data']['batch_size'], True, cfg['data']['num_workers'], cfg['data']['pin_memory'])
    val_loader = make_loader(val_ds, cfg['data']['batch_size'], False, cfg['data']['num_workers'], cfg['data']['pin_memory'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabularTransformerEncoder(num_features=len(schema.feature_cols), **cfg['model']).to(device)
    head = SharedRegressionHead(cfg['model']['d_model']).to(device)
    optimizer = build_optimizer(list(model.parameters()) + list(head.parameters()), cfg['optimizer'])
    trainer_cfg = PretrainTrainerConfig(
        epochs=cfg['pretraining']['epochs'],
        amp=cfg['pretraining']['amp'] and device == 'cuda',
        amp_dtype=cfg['pretraining']['amp_dtype'],
        grad_accum_steps=cfg['pretraining']['grad_accum_steps'],
        clip_grad_norm=cfg['pretraining']['clip_grad_norm'],
        run_dir=str(run_dir),
    )
    masking_cfg = MaskingConfig(
        mask_ratio=cfg['pretraining']['mask_ratio'],
        mask_token_prob=cfg['pretraining']['mask_token_prob'],
        random_replace_prob=cfg['pretraining']['random_replace_prob'],
        keep_original_prob=cfg['pretraining']['keep_original_prob'],
    )
    trainer = PretrainTrainer(model, head, optimizer, train_loader, val_loader, masking_cfg, trainer_cfg, schema.feature_cols, norm_stats, cfg, logger)
    trainer.fit()


if __name__ == '__main__':
    main()

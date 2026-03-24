#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.finetune_dataset import FinetuneDataset, load_real_table, make_cv_splits, stratified_subsample_indices
from data.loaders import make_loader
from data.preprocessing import NormStats, compute_norm_stats_from_array, normalize_array
from data.schema import infer_feature_schema
from losses.classification import make_ce_loss, make_class_weights
from model.heads import ClassificationHead
from model.transformer import TabularTransformerEncoder
from training.checkpointing import load_checkpoint
from training.trainer_finetune import FinetuneTrainer, FinetuneTrainerConfig
from utils.config import load_yaml_config
from utils.io import ensure_dir, save_json
from utils.logging_utils import make_logger
from utils.seed import set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--real_csv', type=str, default=None)
    ap.add_argument('--ckpt_path', type=str, default=None)
    ap.add_argument('--from_scratch', action='store_true')
    return ap.parse_args()


def set_freeze_mode(model, freeze_mode: str, unfreeze_last_layers: int):
    for p in model.parameters():
        p.requires_grad = False
    if freeze_mode == 'linear':
        return
    if freeze_mode == 'full':
        for p in model.parameters():
            p.requires_grad = True
        return
    if freeze_mode == 'partial':
        for p in model.norm.parameters():
            p.requires_grad = True
        layers = model.encoder.layers
        k = max(0, min(unfreeze_last_layers, len(layers)))
        for i in range(len(layers) - k, len(layers)):
            for p in layers[i].parameters():
                p.requires_grad = True
        for p in model.tokenizer.parameters():
            p.requires_grad = True
        return
    raise ValueError(f'Unknown freeze_mode: {freeze_mode}')


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if args.real_csv is not None:
        cfg['data']['real_csv'] = args.real_csv
    if args.ckpt_path is not None:
        cfg['finetuning']['ckpt_path'] = args.ckpt_path
    if args.from_scratch:
        cfg['finetuning']['from_scratch'] = True

    set_seed(cfg['experiment']['seed'])
    run_name = cfg['experiment']['name'] + ('_scratch' if cfg['finetuning']['from_scratch'] else '_pretrained')
    run_dir = ensure_dir(Path(cfg['paths']['output_root']) / 'runs' / 'finetune' / run_name)
    logger = make_logger(run_dir / 'console.log')

    df = pd.read_csv(cfg['data']['real_csv'])
    schema = infer_feature_schema(
        df,
        sex_col=cfg['data']['sex_col'],
        age_col=cfg['data'].get('age_col', 'age'),
        target_column=cfg['data']['target_column'],
        id_col=cfg['data']['id_col'],
        exclude_cols=cfg['data'].get('exclude_cols', []),
    )
    x, y, df = load_real_table(cfg['data']['real_csv'], schema, cfg['data']['target_column'])
    n_classes = int(np.max(y)) + 1
    splits = make_cv_splits(y, cfg['finetuning']['splits'], cfg['experiment']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        fold_dir = ensure_dir(run_dir / f'fold_{fold}')
        xtr_raw, xva_raw = x[tr_idx], x[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        if cfg['finetuning']['label_frac'] < 1.0:
            rel_keep = stratified_subsample_indices(ytr, cfg['finetuning']['label_frac'], cfg['finetuning']['label_min_per_class'], cfg['experiment']['seed'] + fold)
            xtr_raw = xtr_raw[rel_keep]
            ytr = ytr[rel_keep]

        if not cfg['finetuning']['from_scratch'] and cfg['finetuning']['ckpt_path']:
            ckpt = load_checkpoint(cfg['finetuning']['ckpt_path'])
            mean = ckpt['mean']; std = ckpt['std']
            norm_stats = NormStats(mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32))
        else:
            norm_stats = compute_norm_stats_from_array(xtr_raw)

        xtr = normalize_array(xtr_raw, norm_stats)
        xva = normalize_array(xva_raw, norm_stats)
        train_ds = FinetuneDataset(xtr, ytr)
        val_ds = FinetuneDataset(xva, yva)
        train_loader = make_loader(train_ds, cfg['data']['batch_size'], True, cfg['data']['num_workers'], cfg['data']['pin_memory'])
        val_loader = make_loader(val_ds, cfg['data']['batch_size'], False, cfg['data']['num_workers'], cfg['data']['pin_memory'])

        model = TabularTransformerEncoder(num_features=len(schema.feature_cols), **cfg['model']).to(device)
        head = ClassificationHead(cfg['model']['d_model'], n_classes).to(device)
        if not cfg['finetuning']['from_scratch'] and cfg['finetuning']['ckpt_path']:
            model_ckpt = load_checkpoint(cfg['finetuning']['ckpt_path'], map_location='cpu')
            model.load_state_dict(model_ckpt['model_state'], strict=True)
            set_freeze_mode(model, cfg['finetuning']['freeze_mode'], cfg['finetuning']['unfreeze_last_layers'])
        class_weights = make_class_weights(ytr, n_classes).to(device)
        criterion = make_ce_loss(class_weights)
        params = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.parameters() if p.requires_grad], 'lr': cfg['finetuning']['lr_encoder']},
            {'params': head.parameters(), 'lr': cfg['finetuning']['lr_head']},
        ], weight_decay=cfg['optimizer']['weight_decay'])
        trainer_cfg = FinetuneTrainerConfig(
            epochs=cfg['finetuning']['epochs'],
            patience=cfg['finetuning']['patience'],
            amp=cfg['finetuning']['amp'] and device == 'cuda',
            amp_dtype=cfg['finetuning']['amp_dtype'],
            grad_accum_steps=cfg['finetuning']['grad_accum_steps'],
            clip_grad_norm=cfg['finetuning']['clip_grad_norm'],
            run_dir=str(fold_dir),
        )
        fold_logger = make_logger(fold_dir / 'console.log')
        trainer = FinetuneTrainer(model, head, optimizer, criterion, train_loader, val_loader, trainer_cfg, fold_logger, cfg)
        trainer.fit()
        best = load_checkpoint(fold_dir / 'checkpoints' / 'best_macro_f1.pt')
        fold_results.append({'fold': fold, **best['metrics']})

    save_json({'fold_results': fold_results}, run_dir / 'cv_metrics.json')
    logger.info(f'Completed CV. Results saved to {run_dir / "cv_metrics.json"}')


if __name__ == '__main__':
    main()

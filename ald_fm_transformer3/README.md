# ALD Foundation Model — V3

Transformer-based foundation model for lipidomics pretraining, designed for
large-scale synthetic data from the V11 causal model (500 worlds × 20k samples).

## What's new in V3 vs V2

| | V2 | V3 |
|---|---|---|
| Lipid group embedding | — | ✓ 7 groups matching V11 causal structure |
| Generative head | — | ✓ VAE on [CLS] (reconstruction + KL) |
| Multi-GPU | — | ✓ DDP via torchrun |
| LR schedule | constant 1e-3 | ✓ warmup + cosine decay from 3e-4 |
| Mask ratio | 5% | ✓ 20% |
| Model size | d=192, 4L, 6H | d=256, 6L, 8H |
| Batch size | 16 | 256 per GPU |
| Training data | old causal (wrong distribution) | V11 causal (validated) |

## Pretraining

Single GPU (test):
```bash
python pretrain_transformer.py --config configs/pretrain.yaml
```

Multi-GPU (4 GPUs, one node):
```bash
torchrun --nproc_per_node=4 pretrain_transformer.py --config configs/pretrain.yaml
```

On Snellius (SLURM):
```bash
sbatch slurm_pretrain.sh          # 1 node × 4 GPUs
sbatch slurm_pretrain_2node.sh    # 2 nodes × 4 GPUs = 8 GPUs
```

## Fine-tuning

```bash
# From pretrained checkpoint
python finetune_transformer.py --config configs/finetune.yaml \
    --ckpt_path outputs/runs/pretrain/.../checkpoints/best.pt

# From scratch (baseline)
python finetune_transformer.py --config configs/finetune.yaml --from_scratch
```

## Architecture

```
Input: (batch, num_features)  — z-scored lipid values + sex (binary)

Tokenizer:
  token[i] = ValueMLP(x[i]) + FeatureEmb(i) + GroupEmb(group[i])
  [CLS] prepended → sequence length = num_features + 1

Encoder:
  6-layer TransformerEncoder  d=256, 8 heads, FFN=512

Pretraining heads:
  1. SharedRegressionHead  → predict masked lipid values (MLM loss)
  2. VAEBottleneck on [CLS] → (mu, logvar) → z → reconstruct all features

Loss:
  total = masked_MSE + λ_recon × (recon_MSE + β_kl × KL)
  λ_recon=0.5, β_kl=1e-3

Fine-tuning:
  [CLS] → ClassificationHead → 5 severity classes
```

## Group IDs (lipid class embedding)

| ID | Class | Example columns |
|----|-------|-----------------|
| 0 | VLCFA_LPC | LPC_26:0, 1_acyl_LPC_*, LPE_* |
| 1 | PC_PE | PC_38:4, PE_36:2 |
| 2 | PLASMALOGEN | PC_P_38:2, PE_O_36:1, DG_P_* |
| 3 | CERAMIDE | Cer_d42:1, HexCer_*, Hex2Cer_* |
| 4 | SM | SM_d42:1, SM4_* |
| 5 | TG_CE | TG_54:3, CE_24:1, DG_36:2 |
| 6 | OTHER | sex, PI_*, PS_*, PG_* |

## Generative usage

```python
# Load pretrained VAE decoder
vae.eval()
with torch.no_grad():
    synthetic = vae.generate(n_samples=1000, device=device)  # (1000, num_features)
```

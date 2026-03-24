# ALD Transformer Foundation Model

This project implements a transformer-based foundation model for lipidomics data.

## Overview

* Pretraining: self-supervised learning on synthetic datasets (copula, CTGAN/TVAE, Bayesian, causal)
* Fine-tuning: classification on real ALD severity dataset (5 classes)
* Architecture: tabular transformer with one feature per token

---

## Project Structure

configs/ → experiment configurations
data/ → dataset loading & preprocessing
model/ → transformer + embeddings
losses/ → loss functions
training/ → training loops
evaluation/ → metrics & evaluation
outputs/ → checkpoints, logs, results

Main scripts:

* pretrain_transformer.py
* finetune_transformer.py

---

## Data Structure

Example:

data/
synthetic/
copula/
ctgan_tvae/
bayesian/
causal/
real/
ald_dataset.csv

---

## Pretraining

Example:

python pretrain_transformer.py 
--input_folders copula ctgan_tvae

---

## Fine-tuning

From scratch:

python finetune_transformer.py

With pretrained model:

python finetune_transformer.py 
--checkpoint path/to/checkpoint.pt

---

## Outputs

outputs/
runs/
checkpoints/
logs/

---

## Notes

* One feature = one token
* Continuous values (no discretization)
* Masked feature prediction (BERT-style)
* Shared prediction head

# sgRNA-TAC Architecture

## Overview

sgRNA-TAC couples DNABERT encodings with a TripletAttention CNN classifier to predict off-target activity in CRISPR-Cas9 experiments.

## Data Flow

1. `data/PKD.csv` is loaded via `sgrna_tac.data.preprocessing.load_and_preprocess_pkd`.
2. The cleaned dataframe is split into train/val/test splits using deterministic seeds.
3. Sequences are encoded with DNABERT embeddings via `sgrna_tac.encoders.dnabert_encoder`.
4. Encoded tensors are wrapped by `OffTargetDataset` and exposed as PyTorch dataloaders.
5. The TripletAttention CNN consumes wild-type and mutated encodings, fusing them with convolutional blocks and an attention head.
6. Metrics, predictions, and checkpoints are written to `results/`.

## Key Modules

- `sgrna_tac.config`: YAML parsing and validation
- `sgrna_tac.data`: preprocessing helpers and PyTorch dataset abstractions
- `sgrna_tac.encoders`: DNABERT-based tokenization and embedding
- `sgrna_tac.models`: TripletAttention network definition
- `sgrna_tac.training`: end-to-end training/evaluation pipeline
- `train.py`: CLI wrapper for experiments

## Reproducibility

- Random seeds are fixed for train/val/test splits.
- Hyperparameters are centralized under `configs/default.yaml`.
- Results directory structure is deterministic to support downstream analysis and manuscript figures.


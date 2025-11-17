"""PyTorch dataset abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class OffTargetDataset(Dataset):
    """TensorDataset wrapper for WT/mutated embeddings."""

    def __init__(self, wt_encoded: torch.Tensor, mutated_encoded: torch.Tensor, labels: torch.Tensor):
        if wt_encoded.shape[0] != mutated_encoded.shape[0] or wt_encoded.shape[0] != labels.shape[0]:
            raise ValueError("Input tensors must have matching batch dimensions")
        self.wt_encoded = wt_encoded
        self.mutated_encoded = mutated_encoded
        self.labels = labels

    def __len__(self) -> int:  # pragma: no cover - torch Dataset API
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.wt_encoded[idx], self.mutated_encoded[idx], self.labels[idx]


def create_dataloader(
    encoded_sequences: Tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = OffTargetDataset(encoded_sequences[0], encoded_sequences[1], labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


__all__ = ["OffTargetDataset", "create_dataloader"]


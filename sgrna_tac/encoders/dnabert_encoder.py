"""DNABERT-based sequence encoder."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def _kmerize(sequence: str, k: int = 6) -> str:
    return " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))


@lru_cache(maxsize=1)
def _load_tokenizer():
    return AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")


@lru_cache(maxsize=1)
def _load_model(device: torch.device):
    return AutoModel.from_pretrained("zhihan1996/DNA_bert_6").to(device)


def encode_batch(sequences: Iterable[str], device: torch.device, dropout_p: float = 0.0) -> torch.Tensor:
    tokenizer = _load_tokenizer()
    model = _load_model(device)
    dropout = torch.nn.Dropout(p=dropout_p)

    kmers = [_kmerize(seq) for seq in sequences]
    encoded = tokenizer(kmers, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**encoded).last_hidden_state
    return dropout(outputs)


def encode_dataframe(
    dataframe,
    *,
    device: torch.device,
    batch_size: int = 16,
    wt_column: str = "WTSequence (WildType)",
    mutated_column: str = "MutatedSequence",
    label_column: str = "Label",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wt_encoded, mutated_encoded = [], []
    labels = torch.tensor(dataframe[label_column].values, dtype=torch.long, device=device)

    with tqdm(total=len(dataframe), desc="Encoding sequences", unit="seq") as progress:
        for start in range(0, len(dataframe), batch_size):
            stop = start + batch_size
            wt_batch = dataframe[wt_column].iloc[start:stop].tolist()
            mut_batch = dataframe[mutated_column].iloc[start:stop].tolist()

            wt_encoded.append(encode_batch(wt_batch, device))
            mutated_encoded.append(encode_batch(mut_batch, device))
            progress.update(len(wt_batch))

    return torch.cat(wt_encoded, dim=0), torch.cat(mutated_encoded, dim=0), labels


__all__ = ["encode_batch", "encode_dataframe"]


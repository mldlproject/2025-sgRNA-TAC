"""TripletAttention CNN classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletAttention(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 768):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        self.value_proj = nn.Linear(embed_dim, hidden_dim)
        self.energy_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x).unsqueeze(2)
        key = self.key_proj(x).unsqueeze(1)
        value = self.value_proj(x)

        energy = self.energy_proj(F.leaky_relu(query + key + value.unsqueeze(1))).squeeze(-1)
        attn_weights = F.softmax(energy, dim=-1)
        return attn_weights @ value


class TripletAttentionCNN(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.wt_cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.mut_cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.fc_combine = nn.Linear(512, embed_dim)
        self.attention = TripletAttention(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 13, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, wt_input: torch.Tensor, mut_input: torch.Tensor) -> torch.Tensor:
        wt_input = wt_input.permute(0, 2, 1)
        mut_input = mut_input.permute(0, 2, 1)

        wt_feature = self.wt_cnn(wt_input)
        mut_feature = self.mut_cnn(mut_input)

        combined = torch.cat([wt_feature, mut_feature], dim=1).transpose(1, 2)
        combined = self.fc_combine(combined)
        attended = self.attention(combined)
        flattened = torch.flatten(attended, start_dim=1)
        return self.classifier(flattened)


__all__ = ["TripletAttention", "TripletAttentionCNN"]


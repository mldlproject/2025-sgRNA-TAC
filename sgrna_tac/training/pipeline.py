"""Training and evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from sgrna_tac.data.dataset import create_dataloader
from sgrna_tac.data.preprocessing import load_and_preprocess_pkd
from sgrna_tac.encoders.dnabert_encoder import encode_dataframe
from sgrna_tac.models.triplet_attention import TripletAttentionCNN

LOGGER = logging.getLogger(__name__)


def split_dataframe(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float):
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, shuffle=True, random_state=42)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, shuffle=True, random_state=42)
    return train_df, val_df, test_df


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    wt_train, mut_train, labels_train = encode_dataframe(train_df, device=device)
    wt_val, mut_val, labels_val = encode_dataframe(val_df, device=device)
    wt_test, mut_test, labels_test = encode_dataframe(test_df, device=device)

    embed_dim = wt_train.shape[2]

    train_loader = create_dataloader((wt_train, mut_train), labels_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader((wt_val, mut_val), labels_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader((wt_test, mut_test), labels_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, embed_dim


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_epochs: int,
    patience: int,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    criterion = nn.CrossEntropyLoss().to(device)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for wt_batch, mut_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            wt_batch, mut_batch, labels = wt_batch.to(device), mut_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(wt_batch, mut_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for wt_batch, mut_batch, labels in val_loader:
                wt_batch, mut_batch, labels = wt_batch.to(device), mut_batch.to(device), labels.to(device)
                outputs = model(wt_batch, mut_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(val_loader.dataset)

        LOGGER.info("Epoch %d | train %.4f | val %.4f", epoch + 1, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOGGER.info("Early stopping triggered at epoch %d", epoch + 1)
                break
        scheduler.step(val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def fine_tune(
    model: nn.Module,
    dataframe: pd.DataFrame,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> nn.Module:
    wt_encoded, mut_encoded, labels = encode_dataframe(dataframe, device=device)
    loader = create_dataloader((wt_encoded, mut_encoded), labels, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for wt_batch, mut_batch, label_batch in loader:
            wt_batch, mut_batch, label_batch = wt_batch.to(device), mut_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            outputs = model(wt_batch, mut_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * label_batch.size(0)
        epoch_loss /= len(loader.dataset)
        LOGGER.info("Fine-tune epoch %d/%d | loss %.4f", epoch + 1, epochs, epoch_loss)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for wt_batch, mut_batch, label_batch in tqdm(loader, desc="Evaluating"):
            wt_batch, mut_batch = wt_batch.to(device), mut_batch.to(device)
            outputs = model(wt_batch, mut_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            probabilities.extend(probs)
            labels.extend(label_batch.cpu().numpy())

    labels_arr = np.array(labels)
    probs_arr = np.array(probabilities)
    preds = (probs_arr > 0.5).astype(int)

    precision_curve, recall_curve, _ = precision_recall_curve(labels_arr, probs_arr)

    metrics = {
        "Accuracy": float(accuracy_score(labels_arr, preds)),
        "Precision": float(precision_score(labels_arr, preds)),
        "Recall": float(recall_score(labels_arr, preds)),
        "F1": float(f1_score(labels_arr, preds)),
        "ROC_AUC": float(roc_auc_score(labels_arr, probs_arr)),
        "PR_AUC": float(auc(recall_curve, precision_curve)),
    }

    tn, fp, fn, tp = confusion_matrix(labels_arr, preds).ravel()
    metrics.update({"Specificity": float(tn / (tn + fp)), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})

    return metrics, labels_arr, probs_arr


def run_pipeline(config, device: torch.device) -> Dict[str, float]:
    dataset_path = config.data.get("dataset_path")
    if not dataset_path:
        raise ValueError("Config missing data.dataset_path")
    df = load_and_preprocess_pkd(dataset_path)

    train_df, val_df, test_df = split_dataframe(df, config.data["train_ratio"], config.data["val_ratio"], config.data["test_ratio"])

    train_loader, val_loader, test_loader, embed_dim = build_dataloaders(
        train_df, val_df, test_df, batch_size=config.training["batch_size"], device=device
    )

    model = TripletAttentionCNN(embed_dim=embed_dim, num_classes=config.model["num_classes"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.training["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.optimizer["scheduler"]["mode"], patience=config.optimizer["scheduler"]["patience"])

    model, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config.training["num_epochs"],
        patience=config.training["patience"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    model = fine_tune(
        model,
        combined_df,
        batch_size=config.training["batch_size"],
        epochs=config.training.get("final_train_epochs", 2),
        device=device,
    )

    metrics, labels, probs = evaluate(model, test_loader, device)
    output_dir = Path(config.paths["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(output_dir / Path(config.paths["metrics_file"]).name, index=False)
    pd.DataFrame({"label": labels, "probability": probs}).to_csv(output_dir / Path(config.paths["test_predictions"]).name, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embed_dim": embed_dim,
            "best_val_loss": best_val_loss,
            "config": config.raw,
        },
        output_dir / Path(config.paths["model_checkpoint"]).name,
    )

    return metrics


__all__ = ["build_dataloaders", "train_model", "fine_tune", "evaluate", "run_pipeline"]


"""
Training script for Stage 1 Pre-Ulcerative Early Warning Model.
Multi-task loss: BCE loss for hazard risk + MSE loss for MFI / SII indices.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.insole_dataset import InsoleTelemetryDataset
from models.stage1.insole_model import InsoleEarlyWarningModel
from training.train_utils import EarlyStopping, save_checkpoint


def train_stage1(
    data_dir: str = "data/raw/insole_telemetry",
    output_dir: str = "checkpoints/stage1",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Stage 1 model on device: {device}")

    # Load dataset
    full_dataset = InsoleTelemetryDataset(data_dir=data_dir, window_size=100, stride=20)
    if len(full_dataset) == 0:
        raise ValueError(f"No samples found in {data_dir}. Run scripts/generate_synthetic_insole.py first!")

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = InsoleEarlyWarningModel(in_channels=18, d_model=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    bce_loss_fn = nn.BCELoss()
    mse_loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, mode="min")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for feats, labels, mfi_gt, sii_gt in train_loader:
            feats = feats.to(device)
            labels = labels.to(device).unsqueeze(1)
            mfi_gt = mfi_gt.to(device)
            sii_gt = sii_gt.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(feats)

            loss_hazard = bce_loss_fn(outputs["hazard_score"], labels)
            loss_mfi = mse_loss_fn(outputs["mfi_pred"], mfi_gt)
            loss_sii = mse_loss_fn(outputs["sii_pred"], sii_gt)

            loss = loss_hazard + 0.1 * loss_mfi + 0.1 * loss_sii
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(feats)

        train_loss /= train_size

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, labels, mfi_gt, sii_gt in val_loader:
                feats = feats.to(device)
                labels = labels.to(device).unsqueeze(1)
                mfi_gt = mfi_gt.to(device)
                sii_gt = sii_gt.to(device).unsqueeze(1)

                outputs = model(feats)
                loss_h = bce_loss_fn(outputs["hazard_score"], labels)
                loss_m = mse_loss_fn(outputs["mfi_pred"], mfi_gt)
                loss_s = mse_loss_fn(outputs["sii_pred"], sii_gt)

                loss = loss_h + 0.1 * loss_m + 0.1 * loss_s
                val_loss += loss.item() * len(feats)

        val_loss /= val_size
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, os.path.join(output_dir, "best_stage1_model.pth"))

        if early_stopping(val_loss):
            print("Early stopping triggered!")
            break

    print("Stage 1 model training complete! [OK]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 Insole Model")
    parser.add_argument("--data_dir", type=str, default="data/raw/insole_telemetry")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_stage1(data_dir=args.data_dir, epochs=args.epochs)

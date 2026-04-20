"""
Training script for the UNet speech enhancement model.

Usage:
    python -m src.training.train \
        --data_dir  data/processed/voicebank_demand \
        --epochs    50 \
        --batch_size 16 \
        --lr        1e-3 \
        --save_dir  models/

The script expects data_dir to contain:
    train/noisy/   train/clean/
    val/noisy/     val/clean/

Each pair of files must share the same filename (e.g. p226_001.wav).
Run src/data/prepare_voicebank.py first to generate this layout.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.data.dataset import VoiceBankDataset
from src.evaluation.metrics import batch_snr


# ── Loss ──────────────────────────────────────────────────────────────────────

class SpectrogramLoss(nn.Module):
    """
    Combined L1 loss on the magnitude spectrogram.
    L1 is preferred over MSE for audio: it penalises large errors less
    aggressively and tends to produce cleaner perceptual results.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred_mask, noisy_spec, clean_spec):
        enhanced = noisy_spec * pred_mask
        return self.l1(enhanced, clean_spec)


# ── Training utilities ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        mask = model(noisy)
        loss = criterion(mask, noisy, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_snr  = 0.0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        mask     = model(noisy)
        loss     = criterion(mask, noisy, clean)
        enhanced = noisy * mask

        total_loss += loss.item()
        total_snr  += batch_snr(enhanced, clean)

    return total_loss / len(loader), total_snr / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet speech enhancement model")
    parser.add_argument("--data_dir",   type=str, default="data/processed/voicebank_demand")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--save_dir",   type=str, default="models/")
    parser.add_argument("--n_fft",      type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--base_ch",    type=int, default=32)
    parser.add_argument("--depth",      type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset = VoiceBankDataset(
        noisy_dir=os.path.join(args.data_dir, "train", "noisy"),
        clean_dir=os.path.join(args.data_dir, "train", "clean"),
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    val_dataset = VoiceBankDataset(
        noisy_dir=os.path.join(args.data_dir, "val", "noisy"),
        clean_dir=os.path.join(args.data_dir, "val", "clean"),
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(in_channels=1, base_channels=args.base_ch, depth=args.depth).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = SpectrogramLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_snr = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_SNR: {val_snr:.2f} dB | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / "unet_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_snr": val_snr,
                "args": vars(args),
            }, ckpt_path)
            print(f"  → Saved best model to {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
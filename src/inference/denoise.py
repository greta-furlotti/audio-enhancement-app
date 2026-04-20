"""
Inference module: denoise an audio file using the trained UNet model.

This replaces the placeholder smoothing-based denoising with the real
deep learning pipeline:

    WAV in → preprocessing → UNet mask → reconstruction → WAV out

The function signature is intentionally kept identical to the original
placeholder so the FastAPI backend (app/backend/main.py) needs no changes.

Usage (standalone):
    python -m src.inference.denoise path/to/noisy.wav

Usage (from backend):
    from src.inference.denoise import denoise_audio
    output_path = denoise_audio("temp/upload.wav")
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

from src.preprocessing.audio import preprocess_for_model, postprocess_from_model, save_audio
from src.models.unet import UNet


# ── Model loading ─────────────────────────────────────────────────────────────

# Default checkpoint path — override via ENHANCEMENT_MODEL_PATH env variable
_DEFAULT_CKPT = os.environ.get(
    "ENHANCEMENT_MODEL_PATH",
    str(Path(__file__).resolve().parents[3] / "audio-enhancement-app" / "models" / "unet_best.pt"),
)

_model_cache: dict = {}   # cache so we don't reload on every request


def _load_model(ckpt_path: str, device: torch.device) -> UNet:
    """
    Load the UNet from a checkpoint file.
    Caches the model in memory so subsequent calls are instant.

    The checkpoint format is the one written by src/training/train.py:
        {
            "model_state_dict": ...,
            "args": { "base_ch": 32, "depth": 4, ... }
        }
    """
    cache_key = (ckpt_path, str(device))
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt_path}\n"
            "Train the model first with: python -m src.training.train\n"
            "Or set ENHANCEMENT_MODEL_PATH to point to a .pt file."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Read architecture hyperparameters saved alongside weights
    saved_args  = ckpt.get("args", {})
    base_ch     = saved_args.get("base_ch", 32)
    depth       = saved_args.get("depth",   4)

    model = UNet(in_channels=1, base_channels=base_ch, depth=depth)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _model_cache[cache_key] = model
    return model


# ── Main inference function ───────────────────────────────────────────────────

def denoise_audio(
    input_path: str,
    output_path: str = None,
    ckpt_path: str = _DEFAULT_CKPT,
    n_fft: int = 512,
    hop_length: int = 128,
    fixed_len: int = 256,
    device: str = None,
) -> str:
    """
    Denoise a WAV file using the trained UNet model.

    Args:
        input_path:  path to the noisy input WAV file
        output_path: where to write the enhanced WAV (default: input_clean.wav)
        ckpt_path:   path to the .pt model checkpoint
        n_fft:       FFT size — must match the value used during training
        hop_length:  STFT hop — must match training
        fixed_len:   spectrogram time frames fed to the model per chunk
                     (files longer than this are processed in overlapping chunks)
        device:      "cuda", "cpu", or None (auto-detect)

    Returns:
        output_path  (str)
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    if output_path is None:
        output_path = input_path.replace(".wav", "_clean.wav")

    # ── Load model ────────────────────────────────────────────────────────────
    model = _load_model(ckpt_path, torch_device)

    # ── Preprocess ────────────────────────────────────────────────────────────
    # Returns magnitude [F, T], phase [F, T], scale, and metadata
    prep = preprocess_for_model(input_path, n_fft=n_fft, hop_length=hop_length)
    magnitude = prep["magnitude"]   # [F, T]  float32, log-compressed, normalised
    phase     = prep["phase"]       # [F, T]  float32, original noisy phase

    F, T = magnitude.shape

    # ── Chunked inference ─────────────────────────────────────────────────────
    # Audio files can be longer than the fixed_len the model was trained on.
    # We process the spectrogram in non-overlapping chunks and concatenate masks.
    # (For short files T <= fixed_len so this reduces to a single pass.)

    n_chunks   = max(1, (T + fixed_len - 1) // fixed_len)   # ceil division
    mask_chunks = []

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * fixed_len
            end   = min(start + fixed_len, T)
            chunk = magnitude[:, start:end]

            # Pad last chunk to fixed_len if needed
            if chunk.shape[1] < fixed_len:
                pad = fixed_len - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="constant")

            # [F, fixed_len] → [1, 1, F, fixed_len] (batch=1, channels=1)
            tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(torch_device)
            mask   = model(tensor)                          # [1, 1, F, fixed_len]
            mask_np = mask.squeeze().cpu().numpy()          # [F, fixed_len]

            # Remove padding from last chunk
            if i == n_chunks - 1:
                actual_len = end - start
                mask_np = mask_np[:, :actual_len]

            mask_chunks.append(mask_np)

    full_mask = np.concatenate(mask_chunks, axis=1)   # [F, T]

    # ── Reconstruct ───────────────────────────────────────────────────────────
    enhanced_audio = postprocess_from_model(full_mask, prep)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_audio(output_path, enhanced_audio, sample_rate=prep["sample_rate"])
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a WAV file using the trained UNet")
    parser.add_argument("input",        type=str, help="Path to noisy input WAV")
    parser.add_argument("--output",     type=str, default=None, help="Output path (default: input_clean.wav)")
    parser.add_argument("--checkpoint", type=str, default=_DEFAULT_CKPT)
    parser.add_argument("--device",     type=str, default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    out = denoise_audio(args.input, output_path=args.output, ckpt_path=args.checkpoint, device=args.device)
    print(f"Enhanced audio saved to: {out}")
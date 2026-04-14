"""
Evaluation metrics for speech enhancement.

Functions here operate on PyTorch tensors (for use during training)
or numpy arrays (for use in evaluation scripts and notebooks).

Metrics implemented:
    - SNR  (Signal-to-Noise Ratio)        — quick, differentiable proxy
    - PESQ (Perceptual Evaluation of Speech Quality) — ITU-T P.862, requires pesq package
    - STOI (Short-Time Objective Intelligibility)    — requires pystoi package

Install optional dependencies:
    pip install pesq pystoi
"""

import numpy as np
import torch


# ── SNR ───────────────────────────────────────────────────────────────────────

def batch_snr(enhanced: torch.Tensor, clean: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Mean SNR over a batch of spectrograms (in dB).
    Operates on tensors — used directly in the training loop.

    Args:
        enhanced: model output  [B, C, F, T]
        clean:    ground truth  [B, C, F, T]
    Returns:
        mean SNR across the batch (float, dB)
    """
    signal_power = clean.pow(2).mean(dim=(-1, -2, -3))
    noise_power  = (enhanced - clean).pow(2).mean(dim=(-1, -2, -3))
    snr          = 10 * torch.log10(signal_power / (noise_power + eps))
    return snr.mean().item()


def snr_numpy(enhanced: np.ndarray, clean: np.ndarray, eps: float = 1e-8) -> float:
    """SNR on numpy arrays (waveforms or spectrograms)."""
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean((enhanced - clean) ** 2)
    return 10 * np.log10(signal_power / (noise_power + eps))


# ── PESQ ──────────────────────────────────────────────────────────────────────

def compute_pesq(clean_wav: np.ndarray, enhanced_wav: np.ndarray, sr: int = 16000) -> float:
    """
    PESQ score (wide-band, range: -0.5 to 4.5, higher is better).
    Requires: pip install pesq

    Args:
        clean_wav:    clean reference waveform (float32, 1D)
        enhanced_wav: enhanced waveform        (float32, 1D)
        sr:           sample rate (8000 or 16000)
    """
    try:
        from pesq import pesq
    except ImportError:
        raise ImportError("Install pesq: pip install pesq")

    mode = "wb" if sr == 16000 else "nb"
    return float(pesq(sr, clean_wav, enhanced_wav, mode))


# ── STOI ──────────────────────────────────────────────────────────────────────

def compute_stoi(clean_wav: np.ndarray, enhanced_wav: np.ndarray, sr: int = 16000) -> float:
    """
    STOI score (range: 0 to 1, higher is better).
    Requires: pip install pystoi

    Args:
        clean_wav:    clean reference waveform (float32, 1D)
        enhanced_wav: enhanced waveform        (float32, 1D)
        sr:           sample rate
    """
    try:
        from pystoi import stoi
    except ImportError:
        raise ImportError("Install pystoi: pip install pystoi")

    return float(stoi(clean_wav, enhanced_wav, sr, extended=False))


# ── Evaluate a full test set ──────────────────────────────────────────────────

def evaluate_file_pair(clean_path: str, enhanced_path: str, sr: int = 16000) -> dict:
    """
    Compute all metrics for a single (clean, enhanced) file pair.
    Returns a dict with keys: snr, pesq, stoi.
    """
    import librosa
    clean_wav,    _ = librosa.load(clean_path,    sr=sr, mono=True)
    enhanced_wav, _ = librosa.load(enhanced_path, sr=sr, mono=True)

    # Align lengths
    min_len = min(len(clean_wav), len(enhanced_wav))
    clean_wav    = clean_wav[:min_len]
    enhanced_wav = enhanced_wav[:min_len]

    results = {"snr": snr_numpy(enhanced_wav, clean_wav)}

    try:
        results["pesq"] = compute_pesq(clean_wav, enhanced_wav, sr)
    except Exception as e:
        results["pesq"] = None
        print(f"PESQ error: {e}")

    try:
        results["stoi"] = compute_stoi(clean_wav, enhanced_wav, sr)
    except Exception as e:
        results["stoi"] = None
        print(f"STOI error: {e}")

    return results
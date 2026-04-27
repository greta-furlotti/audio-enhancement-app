"""
PyTorch Dataset for VoiceBank+DEMAND.

Returns pairs of (noisy_spectrogram, clean_spectrogram) as float32 tensors
with shape [1, F, T] (single channel, freq bins, time frames).

The magnitude spectrogram is log-compressed and normalised to [0, 1]
before being returned. The phase of the noisy signal is stored separately
and used at inference time to reconstruct the waveform via ISTFT.
"""

import os
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class VoiceBankDataset(Dataset):
    """
    Args:
        noisy_dir:   folder containing noisy .wav files
        clean_dir:   folder containing matching clean .wav files
        n_fft:       FFT size (default 512 → 257 freq bins at 16 kHz)
        hop_length:  STFT hop size (default 128 → ~8 ms at 16 kHz)
        sample_rate: expected sample rate of the audio files
        fixed_len:   if set, all spectrograms are padded/truncated to this
                     number of time frames (required for batching)
    """

    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        n_fft: int = 512,
        hop_length: int = 128,
        sample_rate: int = 16000,
        fixed_len: int = 256,
    ):
        self.noisy_dir   = Path(noisy_dir)
        self.clean_dir   = Path(clean_dir)
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.sample_rate = sample_rate
        self.fixed_len   = fixed_len

        self.files = sorted([f.name for f in self.noisy_dir.glob("*.wav")])
        if not self.files:
            raise FileNotFoundError(f"No .wav files found in {noisy_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_spec(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load WAV and return (log_magnitude, phase) as float32 arrays."""
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        stft      = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag       = np.abs(stft).astype(np.float32)
        phase     = np.angle(stft).astype(np.float32)

        # Log compression: reduces dynamic range, improves training stability
        log_mag = np.log1p(mag)
        return log_mag, phase

    def _pad_or_crop(self, spec: np.ndarray) -> np.ndarray:
        """Ensure spectrogram has exactly self.fixed_len time frames."""
        T = spec.shape[1]
        if T >= self.fixed_len:
            return spec[:, :self.fixed_len]
        pad = self.fixed_len - T
        return np.pad(spec, ((0, 0), (0, pad)), mode="constant")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]

        noisy_mag, _ = self._load_spec(self.noisy_dir / fname)
        clean_mag, _ = self._load_spec(self.clean_dir / fname)

        noisy_mag = self._pad_or_crop(noisy_mag)
        clean_mag = self._pad_or_crop(clean_mag)

        # Normalize to [0, 1] using noisy max (same scale for both)
        scale = noisy_mag.max() + 1e-8
        noisy_mag = noisy_mag / scale
        clean_mag = clean_mag / scale

        # Add channel dim: [F, T] → [1, F, T]
        noisy_tensor = torch.from_numpy(noisy_mag).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_mag).unsqueeze(0)

        return noisy_tensor, clean_tensor
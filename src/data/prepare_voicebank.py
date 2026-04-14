"""
VoiceBank+DEMAND dataset preparation.

Downloads the dataset from the University of Edinburgh DataShare and
reorganises it into the layout expected by the training pipeline:

    data/processed/voicebank_demand/
        train/
            noisy/   *.wav
            clean/   *.wav
        val/
            noisy/   *.wav
            clean/   *.wav

Usage:
    python -m src.data.prepare_voicebank
    python -m src.data.prepare_voicebank --raw_dir data/raw --out_dir data/processed/voicebank_demand

The download is ~2.5 GB. Set VOICEBANK_SKIP_DOWNLOAD=1 if you have
already downloaded the archives manually into raw_dir.
"""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


# ── Dataset URLs ──────────────────────────────────────────────────────────────
# Official Edinburgh DataShare links (stable as of 2024)
URLS = {
    "noisy_trainset_28spk": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip",
    "clean_trainset_28spk": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip",
    "noisy_testset":        "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip",
    "clean_testset":        "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip",
}

# Speakers reserved for validation (held out from training)
VAL_SPEAKERS = {"p226", "p287"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress_hook(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    print(f"\r  {pct}%", end="", flush=True)


def download_and_extract(url: str, dest_dir: Path, skip: bool = False):
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / Path(url).name

    if not skip:
        print(f"Downloading {zip_path.name} ...")
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress_hook)
        print()

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def resample_and_copy(src: Path, dst: Path, target_sr: int = 16000):
    """Load, resample to target_sr, write as 16-bit WAV."""
    audio, sr = librosa.load(src, sr=target_sr, mono=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, audio, target_sr, subtype="PCM_16")


def speaker_id(filename: str) -> str:
    """Extract speaker id from VoiceBank filename, e.g. 'p226_001.wav' → 'p226'."""
    return filename.split("_")[0]


# ── Main preparation ──────────────────────────────────────────────────────────

def prepare(raw_dir: Path, out_dir: Path, target_sr: int = 16000, skip_download: bool = False):

    # 1. Download
    if not skip_download:
        for name, url in URLS.items():
            download_and_extract(url, raw_dir / name)
    else:
        print("Skipping download (VOICEBANK_SKIP_DOWNLOAD=1)")

    # 2. Map raw folders
    raw_noisy_train = raw_dir / "noisy_trainset_28spk" / "noisy_trainset_28spk_wav"
    raw_clean_train = raw_dir / "clean_trainset_28spk" / "clean_trainset_28spk_wav"
    raw_noisy_test  = raw_dir / "noisy_testset"        / "noisy_testset_wav"
    raw_clean_test  = raw_dir / "clean_testset"        / "clean_testset_wav"

    for p in [raw_noisy_train, raw_clean_train, raw_noisy_test, raw_clean_test]:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {p}\n"
                "Run without --skip_download, or check your raw_dir."
            )

    # 3. Split train → train / val by speaker
    print("Processing training set ...")
    all_noisy = sorted(raw_noisy_train.glob("*.wav"))

    for noisy_path in tqdm(all_noisy):
        clean_path = raw_clean_train / noisy_path.name
        spk = speaker_id(noisy_path.name)
        split = "val" if spk in VAL_SPEAKERS else "train"

        resample_and_copy(noisy_path, out_dir / split / "noisy" / noisy_path.name, target_sr)
        resample_and_copy(clean_path, out_dir / split / "clean" / noisy_path.name, target_sr)

    # 4. Test set (kept separate — used only for final evaluation)
    print("Processing test set ...")
    all_noisy_test = sorted(raw_noisy_test.glob("*.wav"))

    for noisy_path in tqdm(all_noisy_test):
        clean_path = raw_clean_test / noisy_path.name
        resample_and_copy(noisy_path, out_dir / "test" / "noisy" / noisy_path.name, target_sr)
        resample_and_copy(clean_path, out_dir / "test" / "clean" / noisy_path.name, target_sr)

    # 5. Summary
    n_train = len(list((out_dir / "train" / "noisy").glob("*.wav")))
    n_val   = len(list((out_dir / "val"   / "noisy").glob("*.wav")))
    n_test  = len(list((out_dir / "test"  / "noisy").glob("*.wav")))
    print(f"\nDone. Files: train={n_train}  val={n_val}  test={n_test}")
    print(f"Output: {out_dir.resolve()}")


# ── Dataset class ─────────────────────────────────────────────────────────────
# (used by src/data/dataset.py — kept here for reference and easy import)

def compute_spectrogram(audio: np.ndarray, n_fft: int = 512, hop_length: int = 128):
    """
    Returns magnitude spectrogram and phase.
    Phase is needed to reconstruct audio via ISTFT.
    """
    stft   = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag    = np.abs(stft)
    phase  = np.angle(stft)
    return mag.astype(np.float32), phase.astype(np.float32)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VoiceBank+DEMAND dataset")
    parser.add_argument("--raw_dir",   type=str, default="data/raw")
    parser.add_argument("--out_dir",   type=str, default="data/processed/voicebank_demand")
    parser.add_argument("--sr",        type=int, default=16000, help="Target sample rate")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, use existing archives in raw_dir")
    args = parser.parse_args()

    skip = args.skip_download or os.environ.get("VOICEBANK_SKIP_DOWNLOAD") == "1"
    prepare(Path(args.raw_dir), Path(args.out_dir), target_sr=args.sr, skip_download=skip)
# Audio Enhancement App

> Deep learning-based audio denoising pipeline with a web demo interface.  
> Built as a showcase of applied ML in signal processing, with relevance to hearables, medical audio and real-world noise suppression.

---

## Overview

This project implements an end-to-end system for audio enhancement using deep learning.  
The pipeline covers data ingestion, model training, inference and evaluation — exposed through a lightweight web application.

Current status: **Phase 1 complete — UNet trained on VoiceBank+DEMAND, evaluated on 824 test files.**

---

## Results (Phase 1 — UNet, 27 epochs)

Evaluated on the full VoiceBank+DEMAND test set (824 files, 16 kHz mono).

| Metric | Score | Description |
|--------|-------|-------------|
| **SNR** | 15.88 dB | Signal-to-noise ratio on enhanced waveforms |
| **PESQ** | 2.54 / 4.5 | Perceptual speech quality (ITU-T P.862 wide-band) |
| **STOI** | 0.921 / 1.0 | Short-time objective intelligibility |

A PESQ of 2.54 corresponds to acceptable quality on the MOS scale, consistent with first-generation magnitude-masking models trained for fewer than 30 epochs. STOI of 0.921 indicates that speech intelligibility is well preserved after enhancement.

---

## Project Structure

```
audio-enhancement-app/
│
├── app/
│   ├── backend/            # FastAPI server (REST API for inference)
│   │   ├── main.py
│   │   └── __init__.py
│   └── frontend/           # Streamlit UI (upload, playback, A/B comparison)
│       ├── app.py
│       └── __init__.py
│
├── src/
│   ├── preprocessing/      # STFT, log-compression, normalisation (audio.py)
│   ├── models/             # Model architectures (unet.py)
│   ├── training/           # Training loop, loss, scheduler (train.py)
│   ├── inference/          # Inference + chunked processing (denoise.py)
│   ├── evaluation/         # Metrics and test-set evaluation scripts
│   └── data/               # Dataset class and download/prepare scripts
│
├── notebooks/              # Experiments and prototyping
├── models/                 # Trained model weights (.pt) — not tracked in git
├── data/                   # Raw and processed datasets — not tracked in git
├── temp/                   # Temporary files during inference — not tracked in git
├── requirements.txt
└── README.md
```

---

## Requirements

**Python:** 3.12  
**OS:** Windows 10/11, Linux, macOS

**System dependency (Windows only):**  
PESQ requires a C compiler. Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the "Desktop development with C++" workload before running `pip install pesq`.

**Python dependencies** (`requirements.txt`):

```
fastapi
uvicorn
streamlit
requests
numpy
scipy
librosa
soundfile
torch
torchaudio
tqdm
pesq
pystoi
openpyxl
pandas
```

---

## Installation

```bash
git clone https://github.com/your-username/audio-enhancement-app.git
cd audio-enhancement-app

python -m venv venv
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
```

---

## Full Pipeline — Step by Step

All commands are run from the project root (`audio-enhancement-app/`).

### 1. Prepare the dataset

Downloads VoiceBank+DEMAND (~2.5 GB) and resamples everything to 16 kHz mono.  
Skip the download flag if you already have the archives locally.

```bash
python -m src.data.prepare_voicebank
```

Output layout:
```
data/processed/voicebank_demand/
    train/noisy/   train/clean/    # ~11,500 files
    val/noisy/     val/clean/      # ~400 files (speakers p226, p287)
    test/noisy/    test/clean/     # 824 files
```

### 2. Train the UNet model

```bash
python -m src.training.train \
    --data_dir   data/processed/voicebank_demand \
    --epochs     50 \
    --batch_size 16 \
    --lr         1e-3 \
    --save_dir   models/
```

The best checkpoint (lowest validation loss) is saved automatically to `models/unet_best.pt`.  
Training can be interrupted at any time — resume by lowering `--epochs` to the remaining count.  
Progress is printed each epoch:

```
Epoch  14/50 | train_loss: 0.0058 | val_loss: 0.0056 | val_SNR: 15.68 dB | 432.1s
  → Saved best model to models/unet_best.pt
```

### 3. Run inference on a single file

```bash
python -m src.inference.denoise path/to/noisy.wav --checkpoint models/unet_best.pt
```

Output is saved as `path/to/noisy_clean.wav`.

### 4. Evaluate on the full test set

Runs inference on all 824 test files (skips files already processed) and computes SNR, PESQ, STOI.

```bash
python -m src.evaluation.evaluate_testset \
    --noisy      data/processed/voicebank_demand/test/noisy \
    --clean      data/processed/voicebank_demand/test/clean \
    --out_dir    data/processed/voicebank_demand/test/enhanced \
    --checkpoint models/unet_best.pt
```

Results are saved to:
- `test/enhanced/metrics.csv` — one row per file
- `test/enhanced/metrics.xlsx` — formatted Excel workbook with summary row
- `test/enhanced/summary.json` — mean scores (loaded by the web app)

### 5. Run the web application

Start the backend and frontend in two separate terminals, both from the project root.

**Terminal 1 — backend:**
```bash
python -m uvicorn app.backend.main:api --reload
```
Backend runs at `http://localhost:8000`  
Interactive API docs at `http://localhost:8000/docs`

**Terminal 2 — frontend:**
```bash
streamlit run app/frontend/app.py
```
Frontend runs at `http://localhost:8501`

**Using the app:**
- Upload a `.wav` file
- Listen to the original audio
- Click **Denoise** — enhanced audio appears immediately with a playback player
- Download the enhanced file with the download button
- The bottom section shows the full test set evaluation table (requires step 4 completed)

---

## Model Pipeline

```
Input WAV
    │
    ▼
librosa.load → float32 waveform (16 kHz mono)
    │
    ▼
STFT (n_fft=512, hop=128) → complex spectrogram
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
Magnitude |X|                  Phase ∠X
log1p compression              stored, not modified
normalise to [0, 1]
    │
    ▼
UNet (encoder–decoder, depth=4, base_ch=32)
    │
    ▼
Soft mask in [0, 1]
    │
    ▼
magnitude × mask → enhanced magnitude
    │
recombine with original phase
    │
    ▼
ISTFT → enhanced waveform
    │
    ▼
soundfile.write → output WAV
```

The UNet predicts a soft mask rather than the clean spectrogram directly.  
This is the standard approach for first-generation magnitude-masking models:  
the mask suppresses frequency bins dominated by noise while preserving speech energy.  
Original phase is reused at reconstruction — the human auditory system is relatively insensitive to absolute phase, especially above ~1 kHz.

---

## Training Metrics

Metrics used **during training** (logged each epoch):

| Metric | Role | Why |
|--------|------|-----|
| **L1 loss** on spectrogram | Training objective | Penalises amplitude errors less aggressively than MSE; tends to produce cleaner perceptual results |
| **SNR** on validation set | Monitoring | Fast to compute on tensors; tracks whether the model is actually reducing noise |

SNR during training is computed on log-compressed normalised spectrograms (not raw waveforms), so the absolute value is higher than the waveform SNR reported in the final evaluation.

---

## Evaluation Metrics

Metrics used **after training** on the full test set (waveform-level, computed by `evaluate_testset.py`):

| Metric | Range | Our score | What it measures |
|--------|-------|-----------|-----------------|
| **SNR** | higher = better | 15.88 dB | Power ratio of clean signal to residual noise |
| **PESQ** | -0.5 to 4.5 | 2.54 | Perceptual quality — simulates the human auditory system using Bark-scale filtering and cognitive distortion modelling. ITU-T P.862 standard, used in hearing aid and VoIP product testing |
| **STOI** | 0 to 1 | 0.921 | Intelligibility — measures whether the temporal modulations of speech (especially consonants) are preserved after processing. High SNR with low STOI indicates the model is smoothing consonants |

PESQ and STOI capture different failure modes. A model can reduce background noise well (high SNR) while distorting consonants (low STOI), or sound clean (high PESQ) but with reduced intelligibility. Reporting all three gives a complete picture.

---

## Roadmap

### Phase 1 — Baseline
- [x] Project structure and web app scaffold (FastAPI + Streamlit)
- [x] UNet architecture on log-magnitude spectrogram (PyTorch)
- [x] Training on VoiceBank+DEMAND (27 epochs, best checkpoint)
- [x] Full test set evaluation: SNR 15.88 dB / PESQ 2.54 / STOI 0.921
- [x] Web demo with audio playback and download

### Phase 2 — Deepened training
- [ ] DNS Challenge dataset (Microsoft)
- [ ] DCCRN or SEGAN architecture (phase-aware, from paper)
- [ ] Noise type classification module
- [ ] Experiment tracking (W&B or MLflow)

### Phase 3 — Production-ready
- [ ] Real-time streaming inference
- [ ] Model quantization for edge devices
- [ ] Biomedical signal extension (ECG/EEG denoising)
- [ ] Docker deployment

---

## Dataset

**Phase 1:** [VoiceBank+DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)  
Clean speech (VoiceBank corpus) mixed with real-world noise environments (DEMAND database).  
Standard benchmark in speech enhancement literature; widely used in INTERSPEECH and ICASSP papers.

**Phase 2 (planned):** [DNS Challenge](https://github.com/microsoft/DNS-Challenge)  
Microsoft's large-scale dataset for real-world noise suppression, used in INTERSPEECH competitions.

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Model training | PyTorch / torchaudio | 2.x |
| Audio I/O | Librosa, SoundFile | — |
| Signal processing | NumPy, SciPy | — |
| Backend API | FastAPI + Uvicorn | — |
| Frontend | Streamlit | — |
| Evaluation | pesq, pystoi | — |
| Excel export | openpyxl | — |

---

## Branch Strategy

```
main        ← stable releases only (Phase 1 complete, Phase 2 complete, ...)
  └─ develop        ← integration branch
       └─ feature/Phase-1-baseline   ← current work
       └─ feature/Phase-2-dccrn      ← next
```

Pull requests go `feature/xxx` → `develop`.  
`develop` → `main` only when a full phase is complete and evaluated.

---

## Contributors

- **Greta Furlotti** — ML architecture, signal processing, training pipeline, evaluation
- **Edoardo Ferraro** — Backend, API, application infrastructure

---

## Notes

This project is developed for research and portfolio purposes, demonstrating practical applications of deep learning in audio signal processing.  
The system architecture is designed to generalise beyond speech — potential applications include biomedical signal denoising (ECG, EEG, audiometric signals) and embedded/edge inference for hearables and medical devices.

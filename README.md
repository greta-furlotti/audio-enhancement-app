# Audio Enhancement App

> Deep learning-based audio denoising pipeline with a web demo interface.  
> Built as a showcase of applied ML in signal processing, with relevance to hearables, medical audio, and real-world noise suppression.

---

## Overview

This project implements an end-to-end system for audio enhancement using deep learning.  
The pipeline covers data ingestion, model training, inference, and evaluation — exposed through a lightweight web application.

Current status: **Phase 1 — Baseline pipeline (dummy denoising, UNet in progress)**

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
│   ├── preprocessing/      # STFT, mel spectrogram, normalization
│   ├── models/             # Model architectures (UNet, baseline CNN)
│   ├── training/           # Training loop, loss functions, optimizer config
│   ├── inference/          # Inference logic (denoise.py)
│   ├── evaluation/         # Metrics: PESQ, STOI, SNR
│   └── data/               # Dataset loaders and download scripts
│
├── notebooks/              # Experiments and prototyping
├── models/                 # Saved model weights (.pt) — not tracked in git
├── data/                   # Raw and processed datasets — not tracked in git
├── temp/                   # Temporary files during inference — not tracked in git
├── requirements.txt
└── README.md
```

---

## Roadmap

### Phase 1 — Baseline demo ✅ in progress
- [x] Project structure and web app scaffold (FastAPI + Streamlit)
- [x] Placeholder denoising pipeline (signal smoothing)
- [ ] UNet architecture on spectrogram (PyTorch)
- [ ] Training on VoiceBank+DEMAND dataset
- [ ] PESQ / STOI evaluation

### Phase 2 — Serious training
- [ ] DNS Challenge dataset (Microsoft)
- [ ] DCCRN or SEGAN architecture (from paper)
- [ ] Noise type classification module
- [ ] Model checkpointing and experiment tracking

### Phase 3 — Production-ready
- [ ] Real-time streaming inference
- [ ] Model quantization for edge devices
- [ ] Biomedical signal extension (ECG/EEG denoising)
- [ ] Docker deployment

---

## Installation

```bash
git clone https://github.com/your-username/audio-enhancement-app.git
cd audio-enhancement-app

python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
```

---

## Usage

**1. Start the backend (FastAPI)**

```bash
python -m uvicorn app.backend.main:api --reload
```

Backend runs at `http://localhost:8000`  
API docs available at `http://localhost:8000/docs`

**2. Start the frontend (Streamlit)**

```bash
streamlit run app/frontend/app.py
```

Frontend runs at `http://localhost:8501`

**3. Use the app**

- Upload a `.wav` file
- Play the original audio
- Click **Denoise** to run inference
- Compare original vs enhanced output

---

## Model Pipeline

```
Input WAV
    │
    ▼
STFT → Spectrogram (magnitude + phase)
    │
    ▼
UNet (encoder-decoder with skip connections)
    │
    ▼
Enhanced spectrogram × original phase
    │
    ▼
ISTFT → Output WAV
```

The UNet architecture is well-suited for spectrogram masking: the encoder learns noise patterns,
the decoder reconstructs clean speech, and skip connections preserve fine-grained spectral detail.
The same architecture is widely used in biomedical signal processing (e.g. ECG denoising, MRI reconstruction).

---

## Evaluation Metrics

| Metric | Description | Relevance |
|--------|-------------|-----------|
| **PESQ** | Perceptual Evaluation of Speech Quality | ITU-T standard, used in hearing aid and VoIP testing |
| **STOI** | Short-Time Objective Intelligibility | Measures speech clarity after enhancement |
| **SNR** | Signal-to-Noise Ratio | Baseline measure of noise reduction |

These metrics are the industry standard for evaluating speech enhancement in hearables and medical devices.

---

## Dataset

**Phase 1:** [VoiceBank+DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)  
Clean speech (VoiceBank corpus) mixed with real-world noise (DEMAND database).  
Standard benchmark in speech enhancement literature.

**Phase 2 (planned):** [DNS Challenge](https://github.com/microsoft/DNS-Challenge)  
Microsoft's large-scale dataset for real-world noise suppression, used in INTERSPEECH competitions.

---

## Tech Stack

- **Python 3.12**
- **PyTorch / torchaudio** — model training and inference
- **FastAPI** — REST API backend
- **Streamlit** — interactive web frontend
- **Librosa / SoundFile** — audio I/O and feature extraction
- **NumPy / SciPy** — signal processing utilities

---

## Contributors

- **Greta Furlotti** — ML architecture, signal processing, training pipeline
- **Edoardo Ferraro** — Backend, API, application infrastructure

---

## Notes

This project is developed for research and portfolio purposes,
demonstrating practical applications of deep learning in audio signal processing.
The system architecture is designed to generalize beyond speech —
potential applications include biomedical signal denoising (ECG, EEG, audiometric signals)
and embedded/edge inference for hearables and medical devices.
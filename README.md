# audio-enhancement-app
Real-time speech enhancement using deep learning with a web demo interface.

# Speech Enhancement App

A deep learning-based speech enhancement system with a simple web interface.  
Upload a noisy audio file and listen to the enhanced output in real time.

---

## Overview

This project demonstrates an end-to-end pipeline for audio denoising:

- Audio preprocessing and feature extraction
- Deep learning model for speech enhancement
- Model inference and evaluation
- Web application for interactive usage

The goal is to simulate a real-world AI system for audio processing, similar to those used in hearables and embedded devices.

---

## Features

- Speech denoising using deep learning (PyTorch)
- Signal visualization (waveform & spectrogram)
- A/B comparison (original vs enhanced)
- Modular ML pipeline (train / evaluate / infer)
- Web app interface (FastAPI + Streamlit)

---

## Project Structure
speech-enhancement-app/
│
├── data/ # Raw and processed datasets
├── src/
│ ├── preprocessing/ # Audio processing (STFT, etc.)
│ ├── models/ # Model architectures
│ ├── training/ # Training scripts
│ ├── inference/ # Inference logic
│ └── evaluation/ # Metrics (SNR, STOI, etc.)
│
├── app/
│ ├── backend/ # FastAPI server
│ └── frontend/ # Streamlit UI
│
├── models/ # Trained models (.pt)
├── notebooks/ # Experiments and prototyping
├── requirements.txt
└── README.md


---

## Installation

```bash
git clone https://github.com/your-username/speech-enhancement-app.git
cd speech-enhancement-app

python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

pip install -r requirements.txt


Usage
1. Start backend (FastAPI)
cd app/backend
uvicorn main:app --reload

2. Start frontend (Streamlit)
cd app/frontend
streamlit run app.py

3. Open in browser
http://localhost:8501



Model

The system uses a deep learning model trained on noisy/clean speech pairs.

Pipeline:
Convert audio → spectrogram (STFT)
Apply neural network
Reconstruct waveform (ISTFT)


Evaluation Metrics
Signal-to-Noise Ratio (SNR)
Short-Time Objective Intelligibility (STOI)


Future Improvements
Real-time streaming inference
Model quantization for edge devices
Noise type classification
Support for biomedical signals (ECG/EEG


Tech Stack
Python
PyTorch
FastAPI
Streamlit
NumPy / SciPy / Librosa)

Contributors
Greta Furlotti – ML & Signal Processing
Edoardo Ferraro – Backend & App Development


Notes

This project is intended for educational and research purposes, showcasing practical applications of AI in audio signal processing.
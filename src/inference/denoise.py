import librosa
import soundfile as sf
import numpy as np
import os

def denoise_audio(input_path: str) -> str:
    """
    Dummy denoising (placeholder)
    """
    audio, sr = librosa.load(input_path, sr=None)

    # fake denoising
    # smoothing
    denoised = np.convolve(audio, np.ones(5)/5, mode='same')

    output_path = input_path.replace(".wav", "_clean.wav")
    sf.write(output_path, denoised, sr)

    return output_path
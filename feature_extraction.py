import librosa
import numpy as np

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    if len(audio) == 0:
        raise ValueError(f"Empty or unreadable audio file: {file_path}")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc

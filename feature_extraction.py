import librosa
import numpy as np

MAX_LEN = 130   # fixed number of time-frames (padded/truncated)
N_MFCC = 40

def extract_features(file_path, duration=3, offset=0.5, n_mfcc=N_MFCC, max_len=MAX_LEN):
    """
    Extracts MFCC sequence features from an audio file.
    Returns shape: (max_len, n_mfcc) -- a time-series of MFCC frames,
    instead of a single averaged vector.
    """
    audio, sample_rate = librosa.load(file_path, duration=duration, offset=offset)

    if len(audio) == 0:
        raise ValueError(f"Empty or unreadable audio file: {file_path}")

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)  # shape: (n_mfcc, time)
    mfcc = mfcc.T  # shape: (time, n_mfcc)

    # Pad or truncate to a fixed length so all samples have the same shape
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len, :]

    return mfcc   # shape: (max_len, n_mfcc) e.g. (130, 40)
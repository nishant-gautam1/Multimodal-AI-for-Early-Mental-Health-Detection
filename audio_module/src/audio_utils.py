# src/audio_utils.py
import librosa
import numpy as np


def extract_mfcc(file_path, n_mfcc=40, max_len=216, duration=3, offset=0.5):
    """
    Extract MFCC features and pad/trim them to a fixed length.
    Used for LSTM-based audio models.

    Args:
        file_path (str): Path to the audio file (.wav)
        n_mfcc (int): Number of MFCC coefficients
        max_len (int): Maximum time steps (pad/trim to this)
        duration (float): Seconds to load
        offset (float): Skip first fraction of a second (reduces non-speech noise)

    Returns:
        np.array: MFCC feature matrix of shape (n_mfcc, max_len)
    """
    try:
        y, sr = librosa.load(file_path, duration=duration, offset=offset)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Pad or trim
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc

    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {file_path}: {e}")
        return None

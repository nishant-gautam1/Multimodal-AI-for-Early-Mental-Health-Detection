import librosa
import numpy as np

def load_audio(path, sr=16000, duration=None, offset=0.0):
    """
    Load an audio file with librosa.
    Returns: (y, sr) or (None, None) on failure
    """
    try:
        y, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)
        return y, sr
    except Exception as e:
        print(f"[load_audio] Error loading {path}: {e}")
        return None, None

def pre_emphasis(signal, coef=0.97):
    """
    Apply pre-emphasis filter to the signal.
    """
    if signal is None:
        return None
    return np.append(signal[0], signal[1:] - coef * signal[:-1])

def trim_silence(signal, top_db=20):
    """
    Trim leading and trailing silence from the signal.
    """
    if signal is None:
        return None
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return trimmed

def normalize_audio(signal):
    """
    Normalize audio to -1..1 range.
    """
    if signal is None:
        return None
    max_abs = np.max(np.abs(signal))
    if max_abs == 0:
        return signal
    return signal / max_abs

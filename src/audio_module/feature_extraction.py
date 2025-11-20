# src/audio_module/feature_extraction.py

import os
import numpy as np
import librosa

try:
    from .preprocessing import load_audio, pre_emphasis, trim_silence, normalize_audio
except (ImportError, SystemError):
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.audio_module.preprocessing import load_audio, pre_emphasis, trim_silence, normalize_audio




def extract_mfcc_from_file(file_path, n_mfcc=40, max_len=216, sr=16000, duration=3.0, offset=0.5):
    """
    Load an audio file, preprocess and extract MFCC features.
    Returns an array shaped (max_len, n_mfcc) suitable for LSTM input (timesteps, features).
    """
    y, _sr = load_audio(file_path, sr=sr, duration=duration, offset=offset)
    if y is None:
        return None

    # Preprocessing
    y = pre_emphasis(y)
    y = trim_silence(y)
    y = normalize_audio(y)

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    except Exception as e:
        print(f"[extract_mfcc_from_file] Error extracting MFCC for {file_path}: {e}")
        return None

    # mfcc shape is (n_mfcc, frames). We want (max_len, n_mfcc) for LSTM (timesteps, features)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # transpose to (timesteps, features)
    mfcc = mfcc.T.astype(np.float32)
    return mfcc

def load_dataset_from_ravdess_tess(ravdess_path, tess_path, ravdess_emotions, tess_emotions, map_to_mental_health, max_len=216, n_mfcc=40):
    """
    Walk RAVDESS and TESS dataset folders and extract MFCCs and labels.
    Returns X (list of arrays shaped (max_len, n_mfcc)) and y (labels).
    """
    X, y = [], []

    # RAVDESS: files are in actor folders
    if os.path.isdir(ravdess_path):
        for actor in os.listdir(ravdess_path):
            actor_folder = os.path.join(ravdess_path, actor)
            if not os.path.isdir(actor_folder):
                continue
            for file in os.listdir(actor_folder):
                if not file.lower().endswith(".wav"):
                    continue
                try:
                    emotion_code = file.split("-")[2]
                    emotion = ravdess_emotions.get(emotion_code)
                    mental_state = map_to_mental_health(emotion) if emotion is not None else None
                    if mental_state:
                        fp = os.path.join(actor_folder, file)
                        mfcc = extract_mfcc_from_file(fp, n_mfcc=n_mfcc, max_len=max_len)
                        if mfcc is not None:
                            X.append(mfcc)
                            y.append(mental_state)
                except Exception as e:
                    print(f"[RAVDESS] Error parsing file {file}: {e}")

    # TESS: files can be in nested folders or root
    if os.path.isdir(tess_path):
        for root, _, files in os.walk(tess_path):
            for file in files:
                if not file.lower().endswith(".wav"):
                    continue
                fname = file.lower()
                for emotion in tess_emotions:
                    if emotion in fname:
                        mental_state = map_to_mental_health(emotion)
                        if mental_state:
                            fp = os.path.join(root, file)
                            mfcc = extract_mfcc_from_file(fp, n_mfcc=n_mfcc, max_len=max_len)
                            if mfcc is not None:
                                X.append(mfcc)
                                y.append(mental_state)
                        break

    return X, y

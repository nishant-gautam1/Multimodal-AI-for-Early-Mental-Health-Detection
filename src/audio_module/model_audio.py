# src/audio_module/model_audio.py
"""
Audio LSTM model definition.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_audio_lstm(input_shape, num_classes, dropout=0.3):
    """
    Build and compile an LSTM-based model for audio features.

    Args:
        input_shape: tuple (timesteps, features) e.g. (216, 40)
        num_classes: int, number of output classes
        dropout: float, dropout rate

    Returns:
        compiled Keras model
    """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(64))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

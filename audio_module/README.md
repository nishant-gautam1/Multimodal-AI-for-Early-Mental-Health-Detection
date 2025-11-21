**AI for Early Mental Health Detection – Audio Classification Model**

This repository contains the audio-based component of the AI for Early Mental Health Detection system.
The objective of this module is to analyze short speech recordings and classify them into three mental health-related states:  
- Normal
- Stressed
- Depressed

The model is built using MFCC feature extraction and a deep learning LSTM architecture.

**1. Project Structure**  

- src/  
  - Contains all Python scripts for preprocessing, training, evaluation, and utility functions.
- flask_app/  
  - Contains the Flask API for real-time audio prediction and the HTML template for uploading audio.
- data/raw/  
  - Place the RAVDESS and TESS datasets here.
- data/processed/  
  - Processed .npz feature file will be saved here.
- model/  
  - Saved trained model and label encoder.

**2. Datasets Used**

This project uses two publicly available emotional speech datasets.

**RAVDESS** – Ryerson Audio-Visual Database of Emotional Speech and Song  
A well-established emotional speech dataset containing neutral, calm, happy, sad, angry, fearful, disgust, and surprised expressions.  
Download:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

**TESS** – Toronto Emotional Speech Set  
A clean emotional speech dataset consisting of seven emotions recorded by two speakers.  
Download:
https://tspace.library.utoronto.ca/handle/1807/24487

Place both datasets in:  
- audio_module/data/raw/RAVDESS/  
- audio_module/data/raw/TESS/

**3. Installation**

- Create a virtual environment:  
  - python -m venv venv  
  - venv\Scripts\activate
- Install dependencies:  
    pip install -r requirements.txt

**4. Preprocessing**

This step extracts MFCC features from all audio files and maps emotions to mental health categories.  
- Run:
  python src/preprocessing.py
- This generates: "data/processed/audio_mental_health_features.npz"

**5. Training the LSTM Model**

- Train the model using:  
  python src/train_audio_model_lstm.py
- The following files will be created:
  - model/audio_lstm_model.h5
  - model/audio_label_encoder.pkl

**6. Evaluating the Model**

Run:  
python src/evaluate.py

This displays accuracy metrics and the confusion matrix.

**7. Running the Flask Application**

- Start the Flask server:  
  python flask_app/app_audio_lstm.py
- Open the URL generated after you run the flask application.

Upload a .wav file to receive predictions for mental health categories.

**8. Requirements**

The project dependencies are listed in requirements.txt:
- numpy
- pandas
- tensorflow
- scikit-learn
- librosa
- flask
- joblib
- matplotlib
- seaborn

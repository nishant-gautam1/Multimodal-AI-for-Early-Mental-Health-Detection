**AI for Early Mental Health Detection – Audio Classification Model**

This repository contains the audio-based component of the AI for Early Mental Health Detection system.
The objective of this module is to analyze short speech recordings and classify them into three mental health-related states:  
- Normal
- Stressed
- Depressed

The model is built using MFCC feature extraction and a deep learning LSTM architecture.

**1. Project Structure**  

project/  
│  
├── src/   
│   ├── audio_utils.py  
│   ├── preprocessing.py  
│   ├── train_audio_model_lstm.py  
│   ├── evaluate.py  
│   └── check_labels.py  
│  
├── flask_app/  
│   ├── app_audio_lstm.py  
│   └── templates/  
│       └── audio_upload.html  
│  
├── data/  
│   ├── raw/          (RAVDESS and TESS datasets placed here)  
│   └── processed/    (processed .npz file generated here)  
│
├── model/  
│   └── audio_lstm_model.h5  
│   └── audio_label_encoder.pkl  
│  
├── requirements.txt  
└── README.md  

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
- audio_modue/data/raw/RAVDESS/  
- audio_module/data/raw/TESS/

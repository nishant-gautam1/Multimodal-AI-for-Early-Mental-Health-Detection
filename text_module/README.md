# Early Mental Health Detection – Text Classification Model

This repository contains the text-based component of the AI for Early Mental Health Detection system. The objective of this module is to analyze short written text and classify it into mental health-related categories.

The trained label encoder contains the following classes:
- Anxiety
- Bipolar  
- Depression   
- Normal   
- Personality disorder   
- Stress  
- Suicidal  

These classes represent different mental-health states identified from the dataset.

**Project Structure**

- src/
  Contains scripts for text preprocessing, training, and evaluation.

- flask_app/
  Contains the Flask API for real-time text prediction and the HTML form.

- data/
  Place the Combined_Data_Expanded.csv dataset here.

- model/
  Contains the trained HuggingFace model folder and label_encoder.pkl.

**Dataset Used**

The text model uses the combined mental-health dataset:

data/Combined_Data_Expanded.csv

The dataset contains conversational text labeled across multiple categories such as:
- Anxiety
- Bipolar
- Depression
- Normal
- Personality disorder
- Stress
- Suicidal

These labels are encoded using label_encoder.pkl.

**Installation**
1. Create a virtual environment:  
   - python -m venv venv
   - venv\Scripts\activate (Windows)
   - source venv/bin/activate (macOS / Linux)
2. Install dependencies:
   pip install -r requirements.txt

**Typical requirements for this project include:**
- flask
- torch
- transformers
- numpy
- pandas
- scikit-learn
- pickle5
- pathlib
- matplotlib
- seaborn
- regex

**Preprocessing**

Run the preprocessing script to clean text, tokenize, pad sequences, and prepare labels:  
python src/preprocessing_text.py

**Running the Text Model**

- Start the Flask application:
  python flask_app/app.py  
- Open the URL shown in the terminal to enter text and receive predictions.  
- The system will output one of the seven mental-health categories.

All predictions are logged inside:  
flask_app/predictions_log.csv

# Multimodal-AI-for-Early-Mental-Health-Detection
**1. Introduction**

This project presents a multimodal deep learning system designed to assist in the early detection of mental health indicators using three human communication modalities: speech, text, and facial expressions.
Each modality captures unique behavioral cues, and the combination of these signals provides a more reliable estimation of emotional patterns that may be associated with early mental health concerns.

The system processes each modality through separate learning pipelines and merges their outputs using a fusion model to provide a single final prediction. This repository includes preprocessing scripts, training modules, and a Streamlit-based application for demonstration.

---

**2. Problem Statement**

Traditional mental health assessment methods rely heavily on manual evaluation and self-reported symptoms. These approaches are subjective and often insufficient for early detection. The lack of automated systems capable of handling real-world multimodal user inputs creates a barrier in providing timely support.

Problem:  
Develop an integrated AI system capable of analyzing audio, text, and facial data to detect early indicators of mental health conditions reliably and efficiently.

---

**3. Objectives**  

**Primary Objectives**

- To build a unified AI system that predicts mental-health conditions from text, audio, and facial images.
- To achieve accurate detection using a combination of NLP, Speech Processing, and Computer Vision.
- To provide an easy-to-use web interface for real-time demonstration.

**Secondary Objectives**

- To preprocess and extract meaningful features from each modality.
- To train robust models for each input type.
- To integrate all components using a Flask-based microservice architecture.
- To design an interpretable system suitable for supervised research demonstration.
  
---
  
**4. Features**

- Speech Emotion Recognition using MFCC-based feature extraction and LSTM networks.  
- Text Emotion/Sentiment Analysis using a sequence-based LSTM classifier.  
- Facial Emotion Recognition using a CNN trained on FER-2013.  
- Multimodal Fusion Model that combines embeddings from audio, text, and image models.  
- Preprocessing pipelines for all modalities.  
- Interactive application interface built using Streamlit.  
- Organized modular project structure suitable for academic or research use.
  
---

**5. Methodology**  

The system architecture consists of three independent models, one for each input modality, followed by a fusion network that integrates all extracted representations.

5.1 Audio Processing

- Audio files are preprocessed using:
   - Noise reduction  
   - Silence trimming  
   - Normalization  
 
- MFCC features are extracted and fed into an LSTM model for emotion classification.

5.2 Text Processing
  - Text data undergoes preprocessing that includes tokenization, truncation/padding, and optional normalization. Each sentence is converted into token IDs and attention masks using a BERT tokenizer. The processed sequences are then fed to a BERT-based sequence classification model for sentiment, mental status, or emotion classification.
      
     - Key steps:
      
     - Tokenization: Split text into BERT tokens and convert to IDs.
     - Padding and Truncation: Ensure all sequences are the same length (max_length=200).
     - Conversion to tensors: Prepare input IDs and attention masks for the model.
     - Feeding to BERT: The preprocessed sequences are input to the model, which predicts the class label.

5.3 Image Processing

- Facial regions are detected from images using classical detection methods.
- Extracted face images are passed to a CNN model trained on emotion data.

5.4 Multimodal Fusion

- Embeddings from audio, text, and image models are concatenated.
- A fully connected neural network merges these embeddings and produces a final prediction.
  
This fusion allows the system to utilize complementary cues from multiple modalities.

---

**6. Technologies Used**
- Machine Learning & Deep Learning
- TensorFlow / Keras
- PyTorch (optional for some preprocessing)
- Transformers (HuggingFace)
  
**Audio Processing:**
- Librosa
- NumPy
- Scikit-learn

**Text Processing:**
- Transformers
- Tokenizers
- NLTK

**Backend / Deployment:**
- Flask
- REST API
- Python

**Frontend:**
- HTML / CSS / JS
  
---

**7. Key Learnings**
- Understanding multimodal deep learning workflows
- Speech feature engineering (MFCC, delta features)
- Face detection and expression analysis using CNNs
- Transformer-based text embeddings
- Deployment of ML models using Flask microservices
- Handling class imbalance & real-world noisy data
- Integration of three different AI pipelines into a unified system
  
---

**8. Future Work**
- Use large-scale multimodal datasets for improved training
- Add real-time webcam and microphone streaming
- Expand labels to cover more nuanced conditions
- Use cross-attention models to combine text + audio + image jointly

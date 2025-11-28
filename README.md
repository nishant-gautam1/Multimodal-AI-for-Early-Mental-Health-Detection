# Multimodal-AI-for-Early-Mental-Health-Detection
**1. Introduction**

This project presents a multimodal deep learning system designed to assist in the early detection of mental health indicators using three human communication modalities: speech, text, and facial expressions.
Each modality captures unique behavioral cues, and the combination of these signals provides a more reliable estimation of emotional patterns that may be associated with early mental health concerns.

The system processes each modality through separate learning pipelines and merges their outputs using a fusion model to provide a single final prediction. This repository includes preprocessing scripts, training modules, and a Streamlit-based application for demonstration.

**2. Features**

- Speech Emotion Recognition using MFCC-based feature extraction and LSTM networks.  
- Text Emotion/Sentiment Analysis using a sequence-based LSTM classifier.  
- Facial Emotion Recognition using a CNN trained on FER-2013.  
- Multimodal Fusion Model that combines embeddings from audio, text, and image models.  
- Preprocessing pipelines for all modalities.  
- Interactive application interface built using Streamlit.  
- Organized modular project structure suitable for academic or research use.  

**3. Methodology**  

The system architecture consists of three independent models, one for each input modality, followed by a fusion network that integrates all extracted representations.

3.1 Audio Processing

- Audio files are preprocessed using:
   - Noise reduction  
   - Silence trimming  
   - Normalization  
 
- MFCC features are extracted and fed into an LSTM model for emotion classification.

3.2 Text Processing
  - Text data undergoes preprocessing that includes tokenization, truncation/padding, and optional normalization. Each sentence is converted into token IDs and attention masks using a BERT tokenizer. The          processed sequences are then fed to a BERT-based sequence classification model for sentiment, mental status, or emotion classification.
      
     - Key steps:
      
     - Tokenization: Split text into BERT tokens and convert to IDs.
     - Padding and Truncation: Ensure all sequences are the same length (max_length=200).
     - Conversion to tensors: Prepare input IDs and attention masks for the model.
     - Feeding to BERT: The preprocessed sequences are input to the model, which predicts the class label.

3.3 Image Processing

- Facial regions are detected from images using classical detection methods.
- Extracted face images are passed to a CNN model trained on emotion data.

3.4 Multimodal Fusion

- Embeddings from audio, text, and image models are concatenated.
- A fully connected neural network merges these embeddings and produces a final prediction.
  
This fusion allows the system to utilize complementary cues from multiple modalities.


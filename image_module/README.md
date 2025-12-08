# Early Mental Health Detection – Image Classification Model

This repository contains the image-based component of the AI for Early Mental Health Detection system.
The objective of this module is to analyze facial expressions from images and classify them into three mental health-related states:  
- Normal
- Stressed
- Depressed

The model is built using CNN (Convolutional Neural Network) architecture for facial emotion recognition.

## 1. Project Structure

```
image_module/
├── src/                    # Python scripts for preprocessing, training, evaluation
│   ├── image_utils.py     # Face detection and image processing utilities
│   ├── preprocessing.py   # Dataset preprocessing pipeline
│   ├── train_image_model.py  # CNN model training
│   └── evaluate.py        # Model evaluation and metrics
├── flask_app/             # Flask API for real-time image prediction
│   ├── app_image.py       # Flask application
│   └── templates/
│       └── image_upload.html  # Web interface for image upload
├── data/                  # Datasets (not included in repo)
│   ├── raw/              # Raw FER-2013 or other datasets
│   └── processed/        # Processed .npz feature files
├── model/                # Saved trained models (not included in repo)
│   ├── image_cnn_model.h5
│   └── image_label_encoder.pkl
├── dataset_research.md   # Comprehensive dataset research
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 2. Datasets

This project recommends using **FER-2013** (Facial Expression Recognition 2013) dataset.

### FER-2013 Dataset
A large-scale facial expression dataset containing 35,887 grayscale images (48x48 pixels).

**Emotions (7 classes):**
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

**Download:**
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

**Emotion Mapping to Mental Health States:**
```python
happy, neutral, surprise → normal
sad, fear, disgust → depressed
angry → stressed
```

For detailed dataset research including alternatives (CK+, AffectNet, RAF-DB), see [dataset_research.md](dataset_research.md).

### Dataset Placement
After downloading, place the dataset in:
```
image_module/data/raw/FER2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── (same structure)
```

## 3. Installation

### Create a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

## 4. Preprocessing

This step processes facial images, detects faces, and maps emotions to mental health categories.

**Run:**
```bash
python src/preprocessing.py
```

**Output:**
- `data/processed/image_mental_health_features.npz`

**What it does:**
- Loads images from FER-2013 dataset
- Detects and crops faces (if needed)
- Resizes images to standard size
- Maps 7 emotions to 3 mental health states
- Saves processed data

## 5. Training the CNN Model

Train the convolutional neural network for facial emotion recognition.

**Run:**
```bash
python src/train_image_model.py
```

**Output:**
- `model/image_cnn_model.h5` - Trained model
- `model/image_label_encoder.pkl` - Label encoder

**Model Architecture:**
- Multiple Conv2D layers with BatchNormalization
- MaxPooling for dimensionality reduction
- Dropout for regularization
- Dense layers for classification
- Softmax output (3 classes)

**Training Configuration:**
- Optimizer: Adam
- Loss: categorical_crossentropy
- Epochs: 50 (configurable)
- Batch size: 32
- Data augmentation: rotation, flip, zoom

## 6. Evaluating the Model

Evaluate model performance on test set.

**Run:**
```bash
python src/evaluate.py
```

**Output:**
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Per-class accuracy metrics

## 7. Running the Flask Application

Start the Flask API server for real-time predictions.

**Run:**
```bash
python flask_app/app_image.py
```

**Access:**
- Open browser: http://localhost:5002
- Upload an image (jpg, png)
- Receive mental health prediction

**API Endpoints:**
- `GET /` - Web interface
- `POST /predict` - Form-based prediction (returns HTML)
- `POST /api/predict` - JSON API endpoint

**Example API Usage:**
```python
import requests

url = "http://localhost:5002/api/predict"
files = {'file': open('face_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
# Output: {"label": "normal", "confidence": 0.87}
```

## 8. Requirements

Key dependencies (see requirements.txt for full list):
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **tensorflow** - Deep learning framework
- **opencv-python** - Image processing and face detection
- **Pillow** - Image loading and manipulation
- **mtcnn** - Face detection (alternative to Haar Cascades)
- **scikit-learn** - Label encoding and metrics
- **Flask** - Web API framework
- **matplotlib, seaborn** - Visualization

## 9. Model Architecture Details

### Custom CNN Architecture:
```
Input (48x48x1 or 224x224x3)
    ↓
Conv2D(32) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(128) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(256) → BatchNorm → ReLU → MaxPool
    ↓
Flatten → Dropout(0.5)
    ↓
Dense(512) → ReLU → Dropout(0.5)
    ↓
Dense(256) → ReLU → Dropout(0.3)
    ↓
Dense(3) → Softmax
```

### Transfer Learning Option:
- Base model: VGG16, ResNet50, or MobileNetV2
- Pre-trained on ImageNet
- Fine-tune top layers
- Custom classification head

## 10. Face Detection Methods

The module supports multiple face detection methods:

1. **Haar Cascades** (OpenCV)
   - Fast, lightweight
   - Good for frontal faces
   - Included in OpenCV

2. **MTCNN** (Multi-task Cascaded CNN)
   - More accurate
   - Detects facial landmarks
   - Better for varied angles

3. **DNN-based** (OpenCV DNN module)
   - Pre-trained deep learning model
   - High accuracy
   - Moderate speed

## 11. Data Augmentation

To improve model generalization, the following augmentations are applied:

- **Rotation:** ±20 degrees
- **Width/Height shift:** 10%
- **Horizontal flip:** Yes
- **Zoom:** ±10%
- **Brightness:** 80-120%

## 12. Performance Tips

### For Better Accuracy:
- Use transfer learning (VGG16/ResNet50)
- Increase training epochs
- Use larger input size (224x224)
- Apply more data augmentation
- Use ensemble of models

### For Faster Training:
- Use smaller input size (48x48)
- Reduce model complexity
- Use GPU acceleration
- Reduce batch size if memory limited

## 13. Integration with Other Modules

This image module is designed to work alongside:
- **Audio Module:** Speech emotion recognition
- **Text Module:** Text sentiment analysis

All three modules output predictions in compatible formats for future multimodal fusion.

## 14. Troubleshooting

**Issue:** Face detection fails
- **Solution:** Try different detection methods (Haar → MTCNN)
- Ensure good image quality and lighting

**Issue:** Low accuracy
- **Solution:** Use transfer learning, increase epochs, add more data

**Issue:** Out of memory during training
- **Solution:** Reduce batch size, use smaller input size

**Issue:** Flask app port conflict
- **Solution:** Change port in app_image.py (default: 5002)

## 15. Future Enhancements

- [ ] Real-time webcam emotion detection
- [ ] Multi-face detection and analysis
- [ ] Temporal emotion tracking (video)
- [ ] Attention mechanisms in CNN
- [ ] Integration with fusion model

## 16. Citation

If using FER-2013 dataset:
```
@inproceedings{goodfellow2013challenges,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
  booktitle={International conference on neural information processing},
  pages={117--124},
  year={2013},
  organization={Springer}
}
```

## 17. License

This code is for educational and research purposes. Please respect the licenses of the datasets used.

---

**Status:** Development version  
**Last Updated:** 2025-12-08  
**Compatibility:** Python 3.8+, TensorFlow 2.13+

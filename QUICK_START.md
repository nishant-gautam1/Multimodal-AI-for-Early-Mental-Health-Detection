# Quick Start Guide - Text & Image Modules

## 📝 Text Module - Production Scripts

### What Was Created
✅ 4 new Python scripts in `text_module/src/`:
- `text_utils.py` - Utility functions
- `text_preprocessing.py` - Data preprocessing
- `train_text_model.py` - BERT training
- `evaluate_text.py` - Model evaluation

### Quick Start
```bash
cd text_module

# 1. Preprocess data
python src/text_preprocessing.py

# 2. Train BERT model
python src/train_text_model.py

# 3. Evaluate model
python src/evaluate_text.py

# 4. Run existing Flask API
python flask_app/text_model.py
```

### Requirements
- Dataset: `data/Combined_Data_Expanded.csv` (not in repo)
- GPU recommended (8GB+ VRAM)
- Training time: ~30-60 min (GPU), ~2-4 hours (CPU)

---

## 🖼️ Image Module - Complete Implementation

### What Was Created
✅ Complete new module with 9 files:
- Dataset research document
- README and requirements
- Image utilities (face detection, preprocessing)
- Preprocessing, training, evaluation scripts
- Flask API with modern web interface

### Quick Start
```bash
cd image_module

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download FER-2013 dataset
# Visit: https://www.kaggle.com/datasets/msambare/fer2013
# Extract to: data/raw/FER2013/

# 3. Preprocess data
python src/preprocessing.py

# 4. Train CNN model
python src/train_image_model.py

# 5. Evaluate model
python src/evaluate.py

# 6. Run Flask API
python flask_app/app_image.py
# Access: http://localhost:5002
```

### Requirements
- Dataset: FER-2013 (35,887 images, ~300MB)
- GPU recommended
- Training time: ~1-2 hours (GPU), ~4-6 hours (CPU)

---

## 🎯 Key Features

### Text Module
- ✅ Production-ready scripts (no Jupyter dependency)
- ✅ BERT fine-tuning with HuggingFace
- ✅ 7-class mental health classification
- ✅ Compatible with existing Flask API

### Image Module
- ✅ Face detection (Haar Cascade + MTCNN)
- ✅ Custom CNN architecture
- ✅ Data augmentation
- ✅ Modern web interface with drag-and-drop
- ✅ REST API on port 5002
- ✅ 3-class mental health classification

---

## 📊 Module Comparison

| Feature | Audio | Text | Image |
|---------|-------|------|-------|
| **Status** | ✅ Complete | ✅ Enhanced | ✅ Complete |
| **Model** | LSTM | BERT | CNN |
| **Input** | .wav files | Text strings | Images |
| **Classes** | 3 | 7 | 3 |
| **Port** | 5000 | 5000 | 5002 |

---

## 🚀 Next Steps

1. **Download Datasets**
   - Text: `Combined_Data_Expanded.csv`
   - Image: FER-2013 from Kaggle

2. **Train Models**
   - Run preprocessing scripts
   - Train models (GPU recommended)
   - Evaluate performance

3. **Test APIs**
   - Run Flask applications
   - Test predictions via web interface
   - Test API endpoints

4. **Future Development**
   - Implement multimodal fusion
   - Create unified API gateway
   - Build Streamlit application

---

## 📁 File Locations

### Text Module
```
text_module/src/
├── text_utils.py              [NEW]
├── text_preprocessing.py      [NEW]
├── train_text_model.py        [NEW]
└── evaluate_text.py           [NEW]
```

### Image Module
```
image_module/
├── dataset_research.md        [NEW]
├── README.md                  [NEW]
├── requirements.txt           [NEW]
├── src/
│   ├── image_utils.py         [NEW]
│   ├── preprocessing.py       [NEW]
│   ├── train_image_model.py   [NEW]
│   └── evaluate.py            [NEW]
└── flask_app/
    ├── app_image.py           [NEW]
    └── templates/
        └── image_upload.html  [NEW]
```

---

## ⚠️ Important Notes

- **No existing files were modified** - All changes are new files
- **Port 5002** for image module (avoids conflict with audio/text on 5000)
- **Datasets not included** - Must download separately
- **GPU recommended** - Significantly faster training
- **Development only** - No deployment configurations

---

## 📚 Documentation

- **Text Module:** See inline comments in each script
- **Image Module:** See [README.md](file:///c:/Users/Srish/Documents/GitHub/Multimodal-AI-for-Early-Mental-Health-Detection/image_module/README.md)
- **Dataset Research:** See [dataset_research.md](file:///c:/Users/Srish/Documents/GitHub/Multimodal-AI-for-Early-Mental-Health-Detection/image_module/dataset_research.md)
- **Complete Walkthrough:** See walkthrough.md artifact

---

**Status:** ✅ Development Complete  
**Ready for:** Dataset acquisition and model training  
**Total Files Created:** 13 new files

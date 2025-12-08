# Image Module - Dataset Research and Recommendations

## Overview
This document provides research on facial emotion recognition datasets suitable for mental health detection through facial expressions.

---

## 📊 Recommended Datasets

### 1. FER-2013 (Facial Expression Recognition 2013)
**⭐ RECOMMENDED FOR THIS PROJECT**

**Description:**
- Large-scale facial expression dataset from Kaggle competition
- Images collected from Google image search
- Grayscale images, 48x48 pixels

**Statistics:**
- **Total images:** 35,887
- **Training set:** 28,709
- **Public test set:** 3,589
- **Private test set:** 3,589

**Emotions (7 classes):**
1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

**Mapping to Mental Health States:**
```python
emotion_to_mental_health = {
    'happy': 'normal',
    'neutral': 'normal',
    'surprise': 'normal',
    'sad': 'depressed',
    'fear': 'depressed',
    'disgust': 'depressed',
    'angry': 'stressed'
}
```

**Pros:**
- ✅ Large dataset size
- ✅ Publicly available
- ✅ Well-documented
- ✅ Widely used in research
- ✅ Challenging (real-world conditions)
- ✅ Free to use

**Cons:**
- ⚠️ Lower resolution (48x48)
- ⚠️ Some labeling noise
- ⚠️ Imbalanced classes

**Download:**
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- Alternative: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

**License:** Public domain

---

### 2. CK+ (Extended Cohn-Kanade Dataset)

**Description:**
- Laboratory-controlled facial expression dataset
- Sequences from neutral to peak expression
- High-quality images

**Statistics:**
- **Subjects:** 123
- **Sequences:** 593
- **Labeled sequences:** 327
- **Image resolution:** 640x490 or 640x480

**Emotions (7 classes):**
1. Anger
2. Contempt
3. Disgust
4. Fear
5. Happy
6. Sadness
7. Surprise

**Pros:**
- ✅ High quality images
- ✅ Controlled environment
- ✅ Temporal sequences available
- ✅ Well-labeled

**Cons:**
- ⚠️ Smaller dataset
- ⚠️ Posed expressions (not natural)
- ⚠️ Requires registration/agreement
- ⚠️ Limited diversity

**Download:**
- Official site: http://www.consortium.ri.cmu.edu/ckagree/
- Requires academic agreement

**License:** Academic use only

---

### 3. AffectNet

**Description:**
- Large-scale facial expression dataset
- Real-world images from internet
- Both categorical and dimensional annotations

**Statistics:**
- **Total images:** 450,000+
- **Manually annotated:** 440,000
- **High-quality subset:** 40,000

**Emotions (8 classes):**
1. Neutral
2. Happy
3. Sad
4. Surprise
5. Fear
6. Disgust
7. Anger
8. Contempt

**Pros:**
- ✅ Very large dataset
- ✅ Real-world images
- ✅ High diversity
- ✅ Both categorical and continuous labels
- ✅ High quality annotations

**Cons:**
- ⚠️ Requires registration
- ⚠️ Large download size
- ⚠️ Academic use only

**Download:**
- Official site: http://mohammadmahoor.com/affectnet/

**License:** Academic/research use

---

### 4. RAF-DB (Real-world Affective Faces Database)

**Description:**
- Real-world facial expressions
- Collected from internet
- Multiple annotators per image

**Statistics:**
- **Total images:** 29,672
- **Training:** 12,271
- **Testing:** 3,068

**Emotions (7 classes):**
1. Surprise
2. Fear
3. Disgust
4. Happiness
5. Sadness
6. Anger
7. Neutral

**Pros:**
- ✅ Real-world images
- ✅ Multiple annotations
- ✅ Good size
- ✅ Diverse

**Cons:**
- ⚠️ Requires registration
- ⚠️ Academic use only

**Download:**
- Official site: http://www.whdeng.cn/raf/model1.html

**License:** Academic use only

---

### 5. JAFFE (Japanese Female Facial Expression)

**Description:**
- Japanese female subjects
- Posed expressions
- High-quality images

**Statistics:**
- **Subjects:** 10 Japanese female models
- **Images:** 213
- **Resolution:** 256x256

**Emotions (7 classes):**
1. Happy
2. Sad
3. Surprise
4. Anger
5. Disgust
6. Fear
7. Neutral

**Pros:**
- ✅ High quality
- ✅ Free to use
- ✅ Well-documented

**Cons:**
- ⚠️ Very small dataset
- ⚠️ Limited diversity (only Japanese females)
- ⚠️ Posed expressions

**Download:**
- https://zenodo.org/record/3451524

**License:** Free for research

---

## 🎯 Recommendation for This Project

### **Primary Choice: FER-2013**

**Reasons:**
1. **Size:** Large enough for deep learning (35K+ images)
2. **Accessibility:** Publicly available, no registration needed
3. **Compatibility:** Aligns well with mental health state mapping
4. **Real-world:** Challenging, realistic conditions
5. **Free:** No licensing restrictions
6. **Community:** Extensive research and baseline models available

### **Secondary Choice: AffectNet**
- Use if you need more data
- Better quality but requires registration
- Larger diversity

---

## 📋 Dataset Comparison Table

| Dataset | Size | Resolution | Emotions | Access | License | Quality |
|---------|------|------------|----------|--------|---------|---------|
| **FER-2013** | 35,887 | 48x48 | 7 | Public | Free | Medium |
| **CK+** | 593 seq | 640x490 | 7 | Registration | Academic | High |
| **AffectNet** | 440,000 | Variable | 8 | Registration | Academic | High |
| **RAF-DB** | 29,672 | Variable | 7 | Registration | Academic | High |
| **JAFFE** | 213 | 256x256 | 7 | Public | Free | High |

---

## 🔄 Emotion to Mental Health Mapping

Based on psychological research and alignment with audio module:

```python
EMOTION_MAPPING = {
    # Normal state
    'happy': 'normal',
    'neutral': 'normal',
    'surprise': 'normal',
    
    # Depressed state
    'sad': 'depressed',
    'fear': 'depressed',
    'disgust': 'depressed',
    
    # Stressed state
    'angry': 'stressed',
    'contempt': 'stressed'  # if available
}
```

**Rationale:**
- Aligns with audio module's 3-class system
- Based on psychological associations
- Maintains consistency across modalities

---

## 📥 Download Instructions

### FER-2013 (Recommended)

**Method 1: Kaggle CLI**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API token (get from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d msambare/fer2013

# Extract
unzip fer2013.zip -d image_module/data/raw/FER2013/
```

**Method 2: Manual Download**
1. Visit: https://www.kaggle.com/datasets/msambare/fer2013
2. Click "Download"
3. Extract to `image_module/data/raw/FER2013/`

**Expected Structure:**
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
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

---

## 🔬 Preprocessing Considerations

### Image Preprocessing Pipeline:
1. **Face Detection:** Use Haar Cascades or MTCNN
2. **Alignment:** Align faces to standard position
3. **Resizing:** Resize to model input size (48x48 or 224x224)
4. **Normalization:** Scale pixel values to [0, 1] or [-1, 1]
5. **Augmentation:** Rotation, flip, brightness adjustment

### Data Augmentation:
```python
augmentation_config = {
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'zoom_range': 0.1,
    'brightness_range': [0.8, 1.2]
}
```

---

## 📚 References

1. **FER-2013:**
   - Goodfellow, I. J., et al. (2013). "Challenges in representation learning: A report on three machine learning contests."

2. **CK+:**
   - Lucey, P., et al. (2010). "The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression."

3. **AffectNet:**
   - Mollahosseini, A., et al. (2017). "AffectNet: A database for facial expression, valence, and arousal computing in the wild."

4. **RAF-DB:**
   - Li, S., et al. (2017). "Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild."

---

## ✅ Next Steps

1. **Download FER-2013** dataset
2. **Implement preprocessing** pipeline
3. **Train baseline CNN** model
4. **Evaluate** on test set
5. **Fine-tune** architecture
6. **Integrate** with Flask API

---

**Last Updated:** 2025-12-08  
**Recommended Dataset:** FER-2013  
**Status:** Ready for implementation

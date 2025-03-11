# Skin Cancer Diseases Diagnosis Using Artificial Intelligence

## Table of Contents
1. Introduction
2. Features
3. Installation
4. Usage
5. Methodology
6. Model Evaluation
7. Results
8. Conclusion
9. References
10. User Guide

---

## 1. Introduction
Skin cancer is one of the most common forms of cancer worldwide. Early detection and diagnosis are crucial in improving treatment outcomes. This project presents a **Convolutional Neural Network (CNN)-based approach** for **automated skin cancer diagnosis** using medical imaging. The model is trained on a labeled dataset to classify different types of skin lesions, providing a reliable AI-driven solution to assist dermatologists in disease diagnosis.

---

## 2. Features
- **AI-Based Diagnosis:** Uses deep learning models for skin cancer classification.
- **Medical Imaging Processing:** Supports various image formats for diagnosis.
- **Data Visualization:** Provides insights into class distribution, age, sex, and localization.
- **Performance Metrics:** Includes accuracy, loss, and confusion matrix analysis.
- **User-Friendly Execution:** Simple command-line interface for training and evaluation.

---

## 3. Installation
### **Dependencies**
Ensure you have Python installed along with the required dependencies:
```bash
pip install numpy pandas tensorflow keras scikit-learn matplotlib seaborn opencv-python imutils efficientnet
```

### **Dataset**
Download the HAM10000 dataset and place it in the appropriate directory.
- [HAM10000 Dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

---

## 4. Usage
### **Run the Model**
To train and evaluate the model, execute the following command:
```bash
python train.py --dataset /path/to/HAM10000_dataset
```
To visualize the results:
```bash
python visualize_results.py
```

---

## 5. Methodology
### **Data Preprocessing**
- Resizing images to **64x64 pixels**.
- Normalizing pixel values to **[0,1]**.
- Handling class imbalance through **oversampling techniques**.

### **Data Visualization**
- **Class Distribution:** Understanding dataset balance.
- **Age & Sex Distribution:** Identifying patient demographics.
- **Localization Analysis:** Mapping lesion locations on the body.

### **Model Architecture**
- CNN with multiple convolutional and pooling layers.
- Batch normalization and dropout for regularization.
- Fully connected layers for final classification.

### **Training**
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Categorical Cross-Entropy
- **Epochs:** Configurable based on performance monitoring

---

## 6. Model Evaluation
### **Metrics Used**
- **Accuracy:** Measures the proportion of correctly classified samples.
- **Loss:** Represents the error rate during classification.
- **Confusion Matrix:** Analyzes per-class classification performance.

---

## 7. Results
**Performance on Test Set:**
- **Test Accuracy:** 87.05%
- **Test Loss:** 0.415

**Training & Validation Metrics:**
![Training Metrics](images/training_metrics.png)

**Confusion Matrix:**
![Confusion Matrix](images/confusion_matrix.png)

---

## 8. Conclusion
This study demonstrates the effectiveness of CNNs in diagnosing skin cancer from medical images. With an accuracy of **87.05%**, the model shows promising potential for clinical applications. Future work may involve:
- **Transfer Learning** to improve performance.
- **Integration with clinical databases** for real-time diagnosis.
- **Enhanced interpretability techniques** for medical explainability.

---

## 9. References
1. Esteva et al., *Dermatologist-level classification of skin cancer with deep neural networks*. Nature, 2017.
2. Gulshan et al., *Development and validation of a deep learning algorithm for detection of diabetic retinopathy*. JAMA, 2016.
3. Rajpurkar et al., *Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning*. arXiv, 2017.

---

## 10. User Guide
### **Steps to Run the Project**
1. **Install Dependencies:**
```bash
pip install imutils efficientnet
```




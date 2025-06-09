# 🧬 Breast Cancer Detection using Machine Learning

This project demonstrates how machine learning techniques can be applied to classify breast tumors as **Malignant (M)** or **Benign (B)** using the Breast Cancer Wisconsin Diagnostic Dataset. The project covers data exploration, preprocessing, model training, hyperparameter tuning, and evaluation.

---

## 📁 Repository Structure

- `breast_cancer_dataset.csv` – Dataset (uploaded locally in this repo)
- `breast_cancer_prediction.py` – Complete Python code with step-by-step ML workflow
- `README.md` – Project documentation

---

## 📊 Dataset Summary

- **Data Source**: [Breast Cancer Wisconsin Diagnostic Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Samples**: 569 instances
- **Features**: 30 numeric features (mean, SE, and worst of radius, texture, etc.)
- **Target Variable**: `diagnosis` – M = Malignant, B = Benign

---

## 🧪 ML Pipeline Overview

### 🔍 1. Data Exploration
- Diagnosis distribution visualization
- Summary statistics
- Histograms and box plots
- Correlation heatmap with top features

### 🧹 2. Data Preprocessing
- Missing value imputation
- Feature scaling with `StandardScaler`
- Label encoding (`M` → 1, `B` → 0)

### 🧠 3. Model Training
Trained and compared five models:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

### 📈 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- ROC Curve and AUC
- Confusion Matrix

### 🔧 5. Hyperparameter Tuning
- `GridSearchCV` applied to fine-tune the **Random Forest** model

### 🔍 6. Feature Importance
- Bar chart showing top 10 influential features

---

## 🚀 Getting Started

### Step 1: Clone the Repo

```bash
git clone https://github.com/Venuraopolsnai/breast-cancer-detection.git
cd breast-cancer-detection

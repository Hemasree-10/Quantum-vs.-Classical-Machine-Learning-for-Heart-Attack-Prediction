# Quantum vs. Classical Machine Learning for Heart Attack Prediction

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Project Overview

This project compares the performance of Quantum Machine Learning (QML) models and Classical Machine Learning (CML) models for predicting heart attacks using a heart disease dataset. The goal is to evaluate whether quantum models can outperform classical models in this medical classification task.

### Models Implemented:
- **Quantum Models**:  
  - Quantum Support Vector Machine (QSVM)  
  - Variational Quantum Classifier (VQC)
- **Classical Models**:  
  - Support Vector Machine (SVM)  
  - Random Forest Classifier  

### Key Findings:

- Classical models (SVM: 85%, Random Forest: 80%) demonstrated superior performance.
- Classical Models Outperform Quantum Models: SVM and Random Forest achieved significantly higher accuracy compared to VQC and QSVC.
- Quantum Models Show Potential: QSVC performed better than VQC, but both quantum models need further optimization to compete with 
   classical models.
- Best Performing Model: *SVM* achieved the highest accuracy (85.71%), making it the most effective model for this dataset. 

---

## 📊 Dataset

### Description:
- **Initial Dataset**: 900 rows with features related to heart health (e.g., age, cholesterol, blood pressure).  
- **Preprocessing**:  
  - Reduced to **300 rows** to work with quantum simulations.  
  - Removed outliers, resulting in **280 rows** for final analysis.  
  - Features were standardized for classification.  

### Dataset Source:
- The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data).  

---

## 🔑 Key Features
- **Quantum Machine Learning**:
  - Implementation of VQC using Qiskit
  - Implementation of QSVM using Qiskit's `QuantumKernel`.    
- **Classical Machine Learning**:  
  - SVM with linear kernel and Random Forest for benchmarking.  
- **Data Preprocessing**:  
  - Standardization, outlier removal, and train-test splitting.  
- **Visualizations**:  
  - Accuracy comparison and confusion matrices.  

---

## ⚙️ Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Steps:
1. Clone the repository:
   ```bash
   (https://github.com/Hemasree-10/Quantum-vs.-Classical-Machine-Learning-for-Heart-Attack-Prediction.git)
   ```
2. Install the required packages:

       pip install -r requirements.txt
3. Download the dataset:

   The dataset is included in the repository under data/.

------

## 🖥️ Usage:
1. Open the Jupyter notebook:

        jupyter notebook quantum_model.ipynb
   
2. Run the notebook cells sequentially to:

- Preprocess the data.

- Train and evaluate VQC, QSVM, SVM, and Random Forest models.

- Generate visualizations (accuracy plots, confusion matrices).
  
-----------

## 📜 License
 
 This project is licensed under the MIT License. See LICENSE for details.

-----------

## 🙏 Acknowledgments
- Dataset: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data).

- Quantum framework: Qiskit.

- Classical ML: scikit-learn.

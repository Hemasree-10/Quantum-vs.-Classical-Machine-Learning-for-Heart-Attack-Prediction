# Quantum vs. Classical Machine Learning for Heart Attack Prediction

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Challenges with Quantum Models](#-challenges-with-quantum-models)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Project Overview

This project compares **quantum machine learning (QML)** and **classical machine learning** models for predicting heart attacks using a **Heart Attack Prediction Dataset**. The dataset initially contained **900 rows**, which was reduced to **300 rows** to work with quantum simulations. After outlier removal, **280 rows** were used for training and testing.

### Models Implemented:
- **Quantum Models**:  
  - Quantum Support Vector Machine (QSVM)  
  - Variational Quantum Classifier (VQC) and SamplerQNN (attempted but faced challenges)  
- **Classical Models**:  
  - Support Vector Machine (SVM)  
  - Random Forest Classifier  

### Key Findings:
- QSVM achieved **60.71% accuracy**, underperforming classical models.  
- Classical models (SVM: 85%, Random Forest: 80%) demonstrated superior performance.  
- Challenges in implementing **VQC/SamplerQNN** highlighted the experimental nature of QML.

---

## 📊 Dataset

### Description:
- **Initial Dataset**: 900 rows with features related to heart health (e.g., age, cholesterol, blood pressure).  
- **Preprocessing**:  
  - Reduced to **300 rows** to work with quantum simulations.  
  - Removed outliers, resulting in **280 rows** for final analysis.  
  - Features were standardized, and labels were binarized for classification.  

### Dataset Source:
- The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data).  

---

## 🔑 Key Features
- **Quantum Machine Learning**:  
  - Implementation of QSVM using Qiskit's `QuantumKernel`.  
  - Attempted implementation of VQC and SamplerQNN.  
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

- Train and evaluate QSVM, SVM, and Random Forest models.

- Generate visualizations (accuracy plots, confusion matrices).
  
-----------

## 🧩 Challenges with Quantum Models
1. VQC/SamplerQNN Implementation:

- Dimension mismatches between predicted and target labels.

- Optimization failures with COBYLA and unstable training.

- Limited quantum circuit expressiveness for the dataset.

2. General QML Challenges:

- Noise in quantum simulations.

- Lack of quantum advantage for structured medical data.
  
-----------

## 📜 License
 
 This project is licensed under the MIT License. See LICENSE for details.

-----------

## 🙏 Acknowledgments
- Dataset: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data).

- Quantum framework: Qiskit.

- Classical ML: scikit-learn.

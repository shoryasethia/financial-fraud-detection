# Credit Card Fraud Detection

A machine learning project implementing multiple approaches to detect fraudulent credit card transactions.

## Project Overview

This project implements various machine learning models to detect credit card fraud using a dataset of transactions. The approaches include traditional ML models, anomaly detection, and deep learning techniques.

## Dataset

The dataset contains credit card transactions made by European cardholders over two days in September 2013. Due to confidentiality issues, the original features have been transformed using PCA.

**Dataset Characteristics:**
- Features: V1-V28 (PCA components), Time, Amount, Class
- Binary Classification: 0 (Normal), 1 (Fraud)
- Highly Imbalanced Dataset
- All numerical features

**Note:** Due to the large size of the dataset (creditcard.csv), it is not included in the repository. You can download it from [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/financial-fraud-detection.git
cd financial-fraud-detection
```

2. Download the dataset
- Download `creditcard.csv` from Kaggle
- Place it in the project root directory

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Models Implemented

1. **Traditional Machine Learning**
   - Random Forest Classifier
   - LightGBM
   - XGBoost

2. **Anomaly Detection**
   - Gaussian Anomaly Detection with Power Transforms
   - GridSearchCV optimization

3. **Deep Learning**
   - Undercomplete Autoencoder
   - Architecture: 128-64-32-16 with BatchNorm
   - Trained on non-fraudulent transactions

## Results

- Autoencoder Performance:
  - F2 Score: 0.962
  - F1 Score: 0.917

## Project Structure

```
financial-fraud-detection/
├── autoencoder.ipynb        # Autoencoder implementation
├── main.ipynb              # Traditional ML models
├── autoencoder.pth         # Saved model weights
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Dependencies

- Python 3.x
- PyTorch
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib
- lightgbm
- xgboost

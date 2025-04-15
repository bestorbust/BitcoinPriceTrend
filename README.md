#  Stock Trend Prediction with GRU Models

Welcome to the Stock Trend Prediction repository! This repository provides a full-stack implementation of a stock trend prediction system using deep learning, specifically **GRU (Gated Recurrent Unit)** neural networks. The system includes data preprocessing, model training (GRU, LSTM, CNN, Transformer), batch prediction, and a web-based prediction interface using Flask.

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Project Description](#project-description)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Usage Examples](#usage-examples)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

##  Introduction

Stock market prediction involves forecasting the future trends of stock prices. Due to the highly dynamic and non-linear nature of the stock market, traditional models often fail to provide accurate predictions. This project leverages the strength of **Recurrent Neural Networks (RNNs)**, particularly **GRU**, which is well-suited for sequential data like time-series stock data.

This repository is a complete solution that:
- Preprocesses stock data from multiple financial platforms
- Trains and evaluates deep learning models
- Provides trend predictions via a web application
- Supports historical batch predictions and real-time forecast

---

## 📋 Project Description

This project includes the following key modules:

- **Data Preprocessing**: Cleans raw data, handles missing values, and performs feature engineering (Daily %, VWAP, Turnover, etc.).
- **Model Training**: Supports GRU, LSTM, CNN, Transformer architectures with early stopping and RMSE evaluation.
- **Batch Prediction**: Predicts stock trends across full datasets and outputs summarized results.
- **Flask Web App**: Lightweight interface for trend prediction based on a user-input date.

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Stock-Trend-Prediction.git
cd Stock-Trend-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Running the Application

### 1. Data Preprocessing
```bash
python train.py --preprocess
```

### 2. Model Training
```bash
python train.py
```

### 3. Batch Trend Prediction
```bash
python trend.py
```

### 4. Launch the Web App
```bash
python main.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser to use the web interface.

---

##  Usage Examples

### Predicting Trend for a Specific Date
```python
from gru_predict import get_gru_trend_for_date

trend = get_gru_trend_for_date("2025-04-10")
print(f"Predicted trend: {trend}")
```

### Training a GRU Model on a Specific Dataset
```python
from train import create_gru, run_pipeline

results = run_pipeline("data/processed_ds1.csv")
```

---

##  File Structure

```
Stock-Trend-Prediction/
├── data/               # Raw and processed stock datasets
├── models/             # Trained model files (.h5)
├── scalers/            # Feature scaler files
├── gru/                # Output predictions and trend results
├── templates/          # HTML templates for Flask web app
├── train.py            # Training pipeline
├── trend.py            # Historical batch prediction
├── gru_predict.py      # Prediction module
├── main.py             # Flask server
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

##  Dependencies

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- Flask
- joblib

Install them with:
```bash
pip install -r requirements.txt
```

---

##  Contributing

Contributions are welcome! If you'd like to improve the labs, add your own examples, or correct any mistakes, please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes with meaningful messages.
4. Test your changes thoroughly.
5. Submit a pull request.

Your contributions will enhance this project and help others in the open-source community.

---

##  License

This project is open-source and licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and share this project under the terms of the license.

# Stock Trend Prediction with GRU Models

This project implements a stock trend prediction system using GRU (Gated Recurrent Unit) neural networks. It includes data preprocessing, model training, and a Flask web interface for predictions.

## Files Description

### 1. gru_predict.py
- **Purpose**: Main prediction module that loads trained GRU models and makes predictions
- **Key Functions**:
  - `load_model_and_scalers()`: Loads trained model and scalers
  - `preprocess_input_data()`: Prepares input data for prediction
  - `predict_trend()`: Makes the actual trend prediction
  - `get_gru_trend_for_date()`: Gets trend for a specific date

### 2. main.py
- **Purpose**: Flask web application for the prediction interface
- **Features**:
  - Single endpoint (`/`) that handles both GET and POST requests
  - Renders an HTML template with prediction results
  - Connects to the GRU prediction module

### 3. train.py
- **Purpose**: Model training pipeline
- **Key Features**:
  - Data cleaning and preprocessing functions
  - Feature engineering for stock data
  - Multiple model architectures (GRU, LSTM, CNN, Transformer)
  - Model training and evaluation
  - Automatic saving of best models and scalers

### 4. trend.py
- **Purpose**: Batch trend prediction for historical data
- **Features**:
  - Processes multiple datasets
  - Generates trend predictions for entire datasets
  - Saves individual and combined trend reports

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Stock-Trend-Prediction.git
   cd Stock-Trend-Prediction
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

2. Install required packages:
   ```bash
   pip install -r requirements.txt

Running the Application
1. Data Preprocessing
Run the preprocessing pipeline to clean and prepare the raw data:

bash
Copy
python train.py --preprocess
2. Model Training
Train all models on the processed datasets:

bash
Copy
python train.py
3. Batch Trend Prediction
Generate trend predictions for historical data:

bash
Copy
python trend.py
4. Web Application
Start the Flask web app for interactive predictions:

bash
Copy
python main.py
Then visit http://localhost:5000 in your browser.

Usage Examples
Getting a trend prediction for a specific date:

python
Copy
from gru_predict import get_gru_trend_for_date

trend = get_gru_trend_for_date("2025-04-10")
print(f"Predicted trend: {trend}")
Training a specific model:

python
Copy
from train import create_gru, run_pipeline

# Train GRU model on a specific dataset
results = run_pipeline("data/processed_ds1.csv")
Dependencies
Python 3.8+

TensorFlow 2.x

scikit-learn

pandas

numpy

Flask

joblib

File Structure
The application expects the following directory structure:

data/: Contains raw and processed datasets

models/: Stores trained model files

scalers/: Contains feature scalers

gru/: Output directory for trend predictions

templates/: Flask HTML templates

License
This project is licensed under the MIT License.

Copy


# 📈 Stock Trend Prediction with GRU Models

This project implements a stock trend prediction system using **GRU (Gated Recurrent Unit)** neural networks. It includes data preprocessing, model training, batch prediction evaluation, and a Flask web interface for interactive predictions.

---

## 📂 File Descriptions

### `gru_predict.py`
- **Purpose**: Main prediction module using trained GRU models.
- **Key Functions**:
  - `load_model_and_scalers()` – Loads GRU model and feature scalers.
  - `preprocess_input_data()` – Processes data into model-ready format.
  - `predict_trend()` – Predicts trend based on input sequence.
  - `get_gru_trend_for_date(date_str)` – Returns trend for a given date.

### `main.py`
- **Purpose**: Flask-based web interface for predictions.
- **Features**:
  - `/` endpoint handles both `GET` and `POST`.
  - Accepts date input and displays predicted trend.
  - Uses GRU model via `gru_predict.py`.

### `train.py`
- **Purpose**: Training pipeline for deep learning models.
- **Features**:
  - Cleans and preprocesses input stock data.
  - Performs feature engineering.
  - Trains LSTM, GRU, CNN, and Transformer models.
  - Saves best-performing model and its scaler.

### `trend.py`
- **Purpose**: Batch trend prediction for historical analysis.
- **Features**:
  - Loads datasets in bulk.
  - Runs predictions over time-series windows.
  - Saves per-dataset and overall trend summaries.

---

## ⚙️ Setup Instructions

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

## 🚀 Running the Application

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
Then open your browser and go to:  
[http://localhost:5000](http://localhost:5000)

---

## 🧪 Usage Examples

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

## 📁 Project Structure

```
Stock-Trend-Prediction/
│
├── data/           # Raw and processed datasets
├── models/         # Saved trained models
├── scalers/        # Feature scalers (StandardScaler)
├── gru/            # Output directory for trend predictions
├── templates/      # Flask HTML templates
├── train.py        # Model training pipeline
├── trend.py        # Batch prediction
├── gru_predict.py  # Model inference module
├── main.py         # Flask app
└── requirements.txt
```

---

## 📌 Dependencies

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- Flask
- joblib

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import losses

# Load the trained GRU model and scalers
def load_model_and_scalers(dataset_name):
    model_path = f"models/"
    
    gru_model = load_model(
        os.path.join(model_path, f"GRU_Model_{dataset_name}_best_model.h5"), 
        custom_objects={'mse': losses.MeanSquaredError()}
    )

    scaler_X = joblib.load(f"scalers/{dataset_name}_scaler_X.pkl")
    scaler_y = joblib.load(f"scalers/{dataset_name}_scaler_y.pkl")

    with open(f"scalers/{dataset_name}_features.json", "r") as f:
        features = json.load(f)
    
    return gru_model, scaler_X, scaler_y, features


# Preprocess the input data for prediction
def preprocess_input_data(data, features, scaler_X):
    data['Prev Close'] = data['Close'].shift(1)
    data['Daily Change %'] = (data['Close'] - data['Prev Close']) / data['Prev Close'] * 100
    data['High-Low %'] = (data['High'] - data['Low']) / data['Low'] * 100
    data['Open-Close %'] = (data['Open'] - data['Close']) / data['Close'] * 100
    data['Turnover'] = data['Close'] * data['Volume']
    data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3

    data = data.dropna()
    data_processed = data[features]
    X_scaled = scaler_X.transform(data_processed.values)

    date_index = data['Date'] if 'Date' in data.columns else pd.date_range(end=pd.Timestamp.today(), periods=len(data))
    
    return X_scaled, date_index


# Predict the trend using GRU
def predict_trend(input_data, gru_model, scaler_y, date_index, dataset_name):
    input_data_rnn = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
    gru_pred = gru_model.predict(input_data_rnn)
    gru_pred = scaler_y.inverse_transform(gru_pred)

    trend = ['No Trend'] + [
        'Increase' if gru_pred[i] > gru_pred[i - 1] else 'Decrease'
        for i in range(1, len(gru_pred))
    ]

    trend_data = pd.DataFrame({
        'Date': pd.to_datetime(date_index[-len(trend):]).dt.strftime('%Y-%m-%d'),
        'Dataset': dataset_name,
        'Predicted Price': gru_pred.flatten(),
        'Trend': trend
    })

    return trend_data


# Main function
def main(dataset_folder, dataset_names):
    all_trends = []

    output_folder = "gru"
    os.makedirs(output_folder, exist_ok=True)

    for dataset_name in dataset_names:
        input_data_path = os.path.join(dataset_folder, f"{dataset_name}.csv")
        if not os.path.exists(input_data_path):
            print(f"Dataset not found: {input_data_path}")
            continue

        print(f"\nProcessing {dataset_name}...")

        gru_model, scaler_X, scaler_y, features = load_model_and_scalers(dataset_name)
        data = pd.read_csv(input_data_path, parse_dates=['Date'])

        X_scaled, date_index = preprocess_input_data(data, features, scaler_X)
        trend_df = predict_trend(X_scaled, gru_model, scaler_y, date_index, dataset_name)

        # Save individual CSV
        output_file = os.path.join(output_folder, f"predicted_trend_{dataset_name}_{datetime.date.today()}.csv")
        trend_df.to_csv(output_file, index=False)
        print(f"Saved trend to {output_file}")

        all_trends.append(trend_df)

    if all_trends:
        combined_df = pd.concat(all_trends, ignore_index=True)
        combined_csv = os.path.join(output_folder, f"combined_trend_all_datasets_{datetime.date.today()}.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"\nAll trends combined and saved as: {combined_csv}")

        
if __name__ == "__main__":
    dataset_folder = "data"
    dataset_names = ["processed_ds1", "processed_ds2", "processed_ds3", "processed_ds4"]
    
    main(dataset_folder, dataset_names)

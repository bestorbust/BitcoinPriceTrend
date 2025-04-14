import os
import pandas as pd
import joblib
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import losses

def load_model_and_scalers(dataset_name):
    model_path = f"models/"
    gru_model = load_model(os.path.join(model_path, f"GRU_Model_{dataset_name}_best_model.h5"),
                        custom_objects={'mse': losses.MeanSquaredError()})
    scaler_X = joblib.load(f"scalers/{dataset_name}_scaler_X.pkl")
    scaler_y = joblib.load(f"scalers/{dataset_name}_scaler_y.pkl")
    with open(f"scalers/{dataset_name}_features.json", "r") as f:
        features = json.load(f)
    return gru_model, scaler_X, scaler_y, features

def preprocess_input_data(data, features, scaler_X):
    data['Prev Close'] = data['Close'].shift(1)
    data['Daily Change %'] = (data['Close'] - data['Prev Close']) / data['Prev Close'] * 100
    data['High-Low %'] = (data['High'] - data['Low']) / data['Low'] * 100
    data['Open-Close %'] = (data['Open'] - data['Close']) / data['Close'] * 100
    data['Turnover'] = data['Close'] * data['Volume']
    data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = data[features]
    data = data.dropna()
    X_scaled = scaler_X.transform(data.values)
    return X_scaled

def predict_trend(X_scaled, model, scaler_y, last_close_price=None):
    X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    pred = model.predict(X_input)
    pred = scaler_y.inverse_transform(pred)

    if len(pred) == 1:
        return "Increase" if pred[0][0] > last_close_price else "Decrease"

    trends = ['No Trend']
    for i in range(1, len(pred)):
        if pred[i] > pred[i - 1]:
            trends.append('Increase')
        else:
            trends.append('Decrease')
    return trends[-1]

def get_gru_trend_for_date(input_date_str):
    trend_file = "gru/combined_trend_all_datasets_2025-04-13.csv"
    input_date = pd.to_datetime(input_date_str)

    if os.path.exists(trend_file):
        df = pd.read_csv(trend_file, parse_dates=["Date"])
        matched_rows = df[df["Date"] == input_date]
        if not matched_rows.empty:
            print(f"Found in CSV: {input_date_str}")
            trend_counts = matched_rows["Trend"].value_counts()
            majority_trend = trend_counts.idxmax()
            return majority_trend

    print(f" Not found in CSV. Predicting using model...")

    dataset_names = ['processed_ds1', 'processed_ds2', 'processed_ds3', 'processed_ds4']
    predicted_trends = []

    for dataset in dataset_names:
        try:
            data_path = f"data/{dataset}.csv"
            raw_data = pd.read_csv(data_path, parse_dates=["Date"])
            model, scaler_X, scaler_y, features = load_model_and_scalers(dataset)

            last_date = raw_data['Date'].max()

            if input_date <= last_date:
                row_index = raw_data.index[raw_data["Date"] == input_date]
                if len(row_index) == 0 or row_index[0] == 0:
                    continue
                idx = row_index[0]
                selected_data = raw_data.iloc[idx-1:idx+1].copy()
                X_scaled = preprocess_input_data(selected_data, features, scaler_X)
                trend = predict_trend(X_scaled, model, scaler_y)
                predicted_trends.append(trend)
            else:
                last_data = raw_data.iloc[-3:].copy()
                last_close = raw_data["Close"].iloc[-1]
                last_data.iloc[-1, raw_data.columns.get_loc("Date")] = input_date
                X_scaled = preprocess_input_data(last_data, features, scaler_X)
                trend = predict_trend(X_scaled, model, scaler_y, last_close_price=last_close)
                predicted_trends.append(trend)

        except Exception as e:
            print(f"Error predicting for {dataset}: {str(e)}")

    if predicted_trends:
        final_trend = max(set(predicted_trends), key=predicted_trends.count)
        return final_trend
    else:
        return "No Trend"

if __name__ == "__main__":
    date_to_check = input("Enter date (YYYY-MM-DD): ").strip()
    result = get_gru_trend_for_date(date_to_check)
    print(f"\nFinal GRU Trend for {date_to_check}: {result}")

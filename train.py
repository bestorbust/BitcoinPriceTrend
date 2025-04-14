import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Conv1D, Flatten, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import json
import os


# ---------- Cleaning Function ----------
def clean_numeric(val):
    if isinstance(val, str):
        val = val.replace(',', '').replace('-', '0')
        if 'K' in val:
            return float(val.replace('K', '')) * 1e3
        elif 'M' in val:
            return float(val.replace('M', '')) * 1e6
        elif 'B' in val:
            return float(val.replace('B', '')) * 1e9
        else:
            return float(val)
    return float(val)

def apply_cleaning(df, cols):
    for col in cols:
        df[col] = df[col].apply(clean_numeric)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(method='ffill')
    return df

# ---------- Feature Engineering ----------
def feature_engineering(df):
    df['Prev Close'] = df['Close'].shift(1)
    df['Daily Change %'] = (df['Close'] - df['Prev Close']) / df['Prev Close'] * 100
    df['High-Low %'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['Open-Close %'] = (df['Open'] - df['Close']) / df['Close'] * 100
    df['Turnover'] = df['Close'] * df['Volume']
    df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df = df.dropna()
    return df

# ---------- Preprocessing Pipelines for all Datasets ----------
def preprocess_and_save():
    # Dataset 1
    df1 = pd.read_csv("data/ds1.csv")
    df1.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Change%']
    df1['Date'] = pd.to_datetime(df1['Date'], format='%d-%m-%Y')
    df1 = df1.sort_values('Date')
    df1['Change%'] = df1['Change%'].str.replace('%', '').astype(float)
    df1 = apply_cleaning(df1, ['Close', 'Open', 'High', 'Low', 'Volume'])
    df1 = feature_engineering(df1)
    df1.to_csv("data/processed_ds1.csv", index=False)

    # Dataset 2
    df2 = pd.read_csv("data/ds2.csv")
    df2.columns = ['Date', 'Close', 'Volume', 'Open', 'High', 'Low']
    df2['Date'] = pd.to_datetime(df2['Date'], format='%m/%d/%Y')
    df2 = df2.sort_values('Date')
    df2 = apply_cleaning(df2, ['Close', 'Open', 'High', 'Low'])
    df2['Volume'] = df2['Volume'].fillna(0)
    df2 = feature_engineering(df2)
    df2.to_csv("data/processed_ds2.csv", index=False)

    # Dataset 3
    df3 = pd.read_csv("data/ds3.csv")
    df3['Date'] = pd.to_datetime(df3['End'], format='%Y-%m-%d')
    df3 = df3[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']]
    df3 = apply_cleaning(df3, ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'])
    df3 = df3.sort_values('Date')
    df3 = feature_engineering(df3)
    df3.to_csv("data/processed_ds3.csv", index=False)

    # Dataset 4
    df4 = pd.read_csv("data/ds4.csv")
    df4['Date'] = pd.to_datetime(df4['End'], format='%d-%m-%Y')
    df4 = df4[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']]
    df4 = apply_cleaning(df4, ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'])
    df4 = df4.sort_values('Date')
    df4 = feature_engineering(df4)
    df4.to_csv("data/processed_ds4.csv", index=False)


# ---------- Model Definitions ----------
def create_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = LayerNormalization()(inputs)
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    output = Dense(1)(x)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def save_plot(fig, dataset_name, model_name, plot_type):
    os.makedirs("plots", exist_ok=True)
    filename = f"{dataset_name}_{model_name}_{plot_type}.png"
    filepath = os.path.join("plots", filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"[+] Plot saved: {filepath}")

# ---------- Training and Evaluation ----------
def build_and_train_model(model_name, model, X_train, X_val, X_test, y_train, y_val, y_test, dataset_name):
    os.makedirs("models", exist_ok=True)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f"models/{model_name}_{dataset_name}_best_model.h5", monitor='val_loss', save_best_only=True, verbose=0)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=32, callbacks=[early_stop,checkpoint], verbose=0)
    

    model.load_weights(f"models/{model_name}_{dataset_name}_best_model.h5")
    test_pred = model.predict(X_test)
    scaler_y = joblib.load(f"scalers/{dataset_name}_scaler_y.pkl")
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_pred_actual = scaler_y.inverse_transform(test_pred).flatten()
    val_mse = mean_squared_error(y_val, model.predict(X_val))
    test_r2 = r2_score(y_test, test_pred)

    # Plot Loss
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    save_plot(fig1, dataset_name=dataset_name, model_name=model_name, plot_type='loss')

    # Plot Actual vs Predicted
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual, label='Actual', color='blue')
    plt.plot(test_pred_actual, label='Predicted', color='red')
    # plt.plot(y_test, label='Actual',color='blue')
    # plt.plot(test_pred, label='Predicted', linestyle='dashed',color='red')
    plt.title(f'{model_name} Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    save_plot(fig2, dataset_name=dataset_name, model_name=model_name, plot_type='actual_vs_predicted')

    return {'model': model_name, 'val_rmse': np.sqrt(val_mse), 'r2_score': test_r2}
def create_sequences(X, y, time_step=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:i + time_step])
        y_seq.append(y[i + time_step])
    return np.array(X_seq), np.array(y_seq)

# ---------- Pipeline ----------
def run_pipeline(csv_path, time_step=30):
    df = pd.read_csv(csv_path)
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    features = ['Prev Close', 'Open', 'High', 'Low', 'VWAP', 'Volume', 'Turnover', 'Daily Change %',
                'High-Low %', 'Open-Close %']
    X = df[features].values
    y = df['Close'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    os.makedirs("scalers", exist_ok=True)
    joblib.dump(scaler_X, f"scalers/{dataset_name}_scaler_X.pkl")
    joblib.dump(scaler_y, f"scalers/{dataset_name}_scaler_y.pkl")
    with open(f"scalers/{dataset_name}_features.json", "w") as f:
        json.dump(features, f)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_step)

    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    input_shape = (time_step, X_train.shape[2])
    results = []
    results.append(build_and_train_model("LSTM_Model", create_lstm(input_shape), X_train, X_val, X_test, y_train, y_val, y_test, dataset_name))
    results.append(build_and_train_model("GRU_Model", create_gru(input_shape), X_train, X_val, X_test, y_train, y_val, y_test, dataset_name))
    results.append(build_and_train_model("CNN_Model", create_cnn(input_shape), X_train, X_val, X_test, y_train, y_val, y_test, dataset_name))
    results.append(build_and_train_model("Transformer_Model", create_transformer(input_shape), X_train, X_val, X_test, y_train, y_val, y_test, dataset_name))

    best_model = min(results, key=lambda x: x['val_rmse'])
    print(f"\n Best Model for {dataset_name}: {best_model['model']} | Validation RMSE: {best_model['val_rmse']:.4f} | R² Score: {best_model['r2_score']:.4f}")

    metrics_file = "metrics_results.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[dataset_name] = results

    with open(metrics_file, "w") as f:
        json.dump(all_results, f, indent=4)

    return results

# ---------- Run Full Pipeline ----------
preprocess_and_save()

dataset_list = ["data/processed_ds1.csv", "data/processed_ds2.csv", "data/processed_ds3.csv", "data/processed_ds4.csv"]
all_results = {}
for dataset in dataset_list:
    print(f"\nRunning pipeline for: {dataset}")
    results = run_pipeline(dataset)
    all_results[dataset] = results

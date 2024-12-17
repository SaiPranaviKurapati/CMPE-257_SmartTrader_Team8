from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
# import matplotlib.pyplot as plt

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period=10):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    return rolling_mean + (rolling_std * 2), rolling_mean, rolling_mean - (rolling_std * 2)

def fetch_and_preprocess_data(ticker, start_date, end_date):
    extended_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    stock_data = yf.download(ticker, start=extended_start_date, end=end_date)

    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}. Please check the date range.")

    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['5-Day_MA'] = stock_data['Close'].rolling(5).mean()
    stock_data['10-Day_MA'] = stock_data['Close'].rolling(10).mean()
    stock_data['5-Day_Volatility'] = stock_data['Close'].rolling(5).std()

    stock_data['Lag_1'] = stock_data['Close'].shift(1)
    stock_data['Lag_2'] = stock_data['Close'].shift(2)
    stock_data['Lag_5'] = stock_data['Close'].shift(5)

    stock_data['RSI'] = calculate_rsi(stock_data)
    stock_data['EMA_10'] = calculate_ema(stock_data)
    stock_data['Bollinger_Upper'], stock_data['Bollinger_Middle'], stock_data['Bollinger_Lower'] = calculate_bollinger_bands(stock_data)

    stock_data.dropna(inplace=True)
    return stock_data

def normalize_data(data, feature_columns, target_column):
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    data[feature_columns] = scaler_features.fit_transform(data[feature_columns])
    data[target_column] = scaler_target.fit_transform(data[[target_column]])
    return data, scaler_features, scaler_target

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    features = [
        'Daily_Return', '5-Day_MA', '10-Day_MA', '5-Day_Volatility',
        'RSI', 'EMA_10', 'Bollinger_Upper', 'Bollinger_Middle',
        'Bollinger_Lower', 'Lag_1', 'Lag_2', 'Lag_5'
    ]
    target = 'Close'
    lookback = 5

    stock_data = fetch_and_preprocess_data("NVDA", start_date, end_date)
    stock_data, scaler_features, scaler_target = normalize_data(stock_data, features, target)

    X_train, X_test, y_train, y_test = train_test_split(
        stock_data[features].values,
        stock_data[target].values,
        test_size=0.2,
        random_state=42
    )

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Train LSTM
    X_lstm, y_lstm = [], []
    for i in range(len(stock_data) - lookback):
        X_lstm.append(stock_data[features].iloc[i:i + lookback].values)
        y_lstm.append(stock_data[target].iloc[i + lookback])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    lstm_model = build_lstm_model((lookback, len(features)))
    lstm_model.fit(
        X_lstm[:-20], y_lstm[:-20],
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )

    with open("random_forest_model.pkl", "wb") as rf_file:
        joblib.dump(rf, rf_file, compress=3)
    with open("xgboost_model.pkl", "wb") as xgb_file:
        joblib.dump(xgb_model, xgb_file, compress=3)
    lstm_model.save("lstm_model.h5")

    with open("scaler_features.pkl", "wb") as sf_file:
        joblib.dump(scaler_features, sf_file, compress=3)
    with open("scaler_target.pkl", "wb") as st_file:
        joblib.dump(scaler_target, st_file, compress=3)

    print("Models and scalers saved successfully!")


def evaluate_model(true, predicted, model_name):
    """
    Function to evaluate model performance and print metrics.
    """
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, predicted)
    r2 = r2_score(true, predicted)

    print(f"Performance Metrics for {model_name}:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print("-" * 40)

# Evaluate Random Forest
rf_predictions = rf.predict(X_test)
evaluate_model(y_test, rf_predictions, "Random Forest")

# Evaluate XGBoost
xgb_predictions = xgb_model.predict(X_test)
evaluate_model(y_test, xgb_predictions, "XGBoost")

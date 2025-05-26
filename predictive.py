#### Predictive Analaytics: Harga Cardano melalui Time Series Forecasting

"""
Dataset Source: https://www.bitget.com/price/cardano/historical-data#download

Predictive Analysis Time Series Forecasting

## 1. Pendahuluan

### Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan model machine learning agar bisa membantu memprediksi harga cardano sesuai dengan analisa. 

Notebook ini berisi implementasi tiga model untuk memprediksi harga Cardano:
- Facebook Prophet (model statistik berbasis dekomposisi)
- XGBoost (model berbasis tree ensemble)
- LSTM (model deep learning untuk sequence data)
"""

"""
## 2. Import Library 
Di bagian ini, kita mengimport library yang diperlukan
"""

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import xgboost as xgb
import tensorflow as tf
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

"""
## 3. Data Understanding
Data Understanding merupakan proses memahami informasi dalam data tersebut.

### 3.1 Data Loading

**Informasi Datasets**

| Jenis | Keterangan |
| ------ | ------ |
| Title | Cardano Price History |
| Source | [Bitget](https://www.bitget.com/price/cardano/historical-data#download) |
"""

df = pd.read_csv('cardano.csv')

"""
## 3.2. Exploratory Data Analysis

Exploratory Data Analysis adalah proses investigasi awal untuk melakukan analisis dari data secara karakteristik, pola, outlier, dan asumsi pada data.

### 3.2.1 EDA - Deskripsi Variabel 
"""

print(df)

"""
Dataset berisi 9 variabel utama:

| Variabel | Deskripsi | Tipe Data |
|----------|-----------|-----------|
| **timeOpen** | Timestamp Unix (milidetik) awal periode | int64 |
| **timeClose** | Timestamp Unix (milidetik) akhir periode | int64 |
| **timeHigh** | Timestamp saat harga tertinggi | int64 |
| **timeLow** | Timestamp saat harga terendah | int64 |
| **priceOpen** | Harga pembukaan Cardano (USD) | float64 |
| **priceHigh** | Harga tertinggi Cardano (USD) | float64 |
| **priceLow** | Harga terendah Cardano (USD) | float64 |
| **priceClose** | Harga penutupan Cardano (USD) | float64 |
| **volume** | Volume perdagangan Cardano | float64 |
"""

# Convert Unix timestamps (milliseconds) to datetime format
df['dateOpen'] = pd.to_datetime(df['timeOpen'], unit='ms')
df['dateClose'] = pd.to_datetime(df['timeClose'], unit='ms')
df['dateHigh'] = pd.to_datetime(df['timeHigh'], unit='ms')
df['dateLow'] = pd.to_datetime(df['timeLow'], unit='ms')

# Display basic info
print("DataFrame shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nDate range:", df['dateOpen'].min(), "to", df['dateOpen'].max())

# Urutkan data berdasarkan tanggal untuk analisis time series
df = df.sort_values('dateOpen')

# Tambahkan fitur turunan untuk analisis
df['day'] = df['dateOpen'].dt.date
df['volatility'] = df['priceHigh'] - df['priceLow']  # Volatilitas harian
df['returnPercentage'] = df['priceClose'].pct_change() * 100  # Return harian dalam persen
df['range_percent'] = (df['priceHigh'] - df['priceLow']) / df['priceOpen'] * 100  # Range harian dalam persen

# Tampilkan statistik deskriptif
print(f"Dataset data range  {df['dateOpen'].min()} to {df['dateClose'].max()}")
print(f"Total data points: {df.shape[0]} daily records")
print(f"Date coverage: {(df['dateClose'].max() - df['dateOpen'].min()).days + 1} days")

# Statistik harga dan volume
print(f"Harga Terendah: {df['priceLow'].min():,.2f} USD / {df['priceLow'].min() * 16410:,.2f} IDR")
print(f"Harga Tertinggi: {df['priceHigh'].max():,.2f} USD / {df['priceHigh'].max() * 16410:,.2f} IDR")
print(f"Volatilitas Tertinggi: {df['volatility'].max():,.2f} USD")
print(f"Return Tertinggi: {df['returnPercentage'].max():,.2f}%")
print(f"Return Terendah: {df['returnPercentage'].min():,.2f}%")

# Hitung Missing Values dan Duplicates
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicates:")
print(df.duplicated().sum())

"""
### 3.2.2 Menangani Missing Values dan Duplicates
"""

# Set visualization style and size
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 7)

# Sort by date for proper time series analysis
df = df.sort_values('dateOpen')

# Add helpful features
df['day'] = df['dateOpen'].dt.date
df['volatility'] = df['priceHigh'] - df['priceLow']  # Daily volatilitas
df['returnPercentage'] = df['priceClose'].pct_change() * 100   # Daily return dalam bentuk persentase
df['range_percent'] = (df['priceHigh'] - df['priceLow']) / df['priceOpen'] * 100  # Daily range in percentage

df = df.dropna()  # Menghapus baris dengan missing values
print(f"\nTotal rows after dropping missing values: {df.shape[0]}")
print(f"Total missing values: {df.isnull().sum().sum()}")

"""
#### 3.2.3 Visualisasi Data Harga dan Tren
"""

plt.figure(figsize=(14, 8))
plt.plot(df['dateOpen'], df['priceClose'], label='Close Price', color='#2070b4', linewidth=1.5)

# Add moving averages
df['MA_7'] = df['priceClose'].rolling(window=7).mean()
df['MA_30'] = df['priceClose'].rolling(window=30).mean()
df['MA_90'] = df['priceClose'].rolling(window=90).mean()

plt.plot(df['dateOpen'], df['MA_7'], label='7-Day MA', color='#f58021', linewidth=1)
plt.plot(df['dateOpen'], df['MA_30'], label='30-Day MA', color='#e6317c', linewidth=1.5)
plt.plot(df['dateOpen'], df['MA_90'], label='90-Day MA', color='#3bbf4e', linewidth=1.8)

plt.title('Cardano (ADA) Price Evolution with Moving Averages', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Conversion rate from USD to IDR (example rate, update as needed)
usd_to_idr = 16410  # (Updated 20 Mei 2025)

# Format y-axis to show Rupiah amounts
def rupiah_format(x, pos):
    # Convert to IDR then format
    x_idr = x * usd_to_idr
    return f'Rp{x_idr:,.0f}'

plt.gca().yaxis.set_major_formatter(FuncFormatter(rupiah_format))
plt.show()

plt.figure(figsize=(14, 8))
plt.plot(df['dateOpen'], df['priceClose'], label='Close Price', color='#2070b4', linewidth=1.5)

# Add Exponential Moving Averages (EMA) (lebih responsif terhadap perubahan harga)
df['EMA_7'] = df['priceClose'].ewm(span=7, adjust=False).mean()
df['EMA_30'] = df['priceClose'].ewm(span=30, adjust=False).mean()
df['EMA_90'] = df['priceClose'].ewm(span=90, adjust=False).mean()

plt.plot(df['dateOpen'], df['EMA_7'], label='7-Day EMA', color='#f58021', linewidth=1)
plt.plot(df['dateOpen'], df['EMA_30'], label='30-Day EMA', color='#e6317c', linewidth=1.5)
plt.plot(df['dateOpen'], df['EMA_90'], label='90-Day EMA', color='#3bbf4e', linewidth=1.8)

plt.title('Cardano (ADA) Price Evolution with Exponential Moving Averages', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (IDR)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.gca().yaxis.set_major_formatter(FuncFormatter(rupiah_format))
plt.show()

"""
#### 4. DATA PREPARATION
"""

# Split data into training (80%) and testing (20%) sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Feature engineering
df['MA_7'] = df['priceClose'].rolling(window=7).mean()
df['MA_30'] = df['priceClose'].rolling(window=30).mean()
df['EMA_7'] = df['priceClose'].ewm(span=7, adjust=False).mean()
df['volatility'] = df['priceHigh'] - df['priceLow']
df['volume_change'] = df['volume'].pct_change()
df['price_change'] = df['priceClose'].pct_change()

# Scale the data for neural networks
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['priceClose'].values.reshape(-1, 1))

"""
#### 5. Modelling

### Prophet Model
"""

# Siapkan data Prophet (memerlukan kolom 'ds' dan 'y')
prophet_df = pd.DataFrame({
    'ds': df['dateOpen'],
    'y': df['priceClose']
})

# Split data menjadi training (80%) dan testing (20%)
train_size = int(len(prophet_df) * 0.8)
train_df = prophet_df.iloc[:train_size]
test_df = prophet_df.iloc[train_size:]

# Buat dan fit model Prophet
prophet_model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05,  # Kelenturan tren
    seasonality_prior_scale=10.0   # Kekuatan musiman
)
prophet_model.fit(train_df)

# Buat future dataframe untuk periode test
# Gunakan jumlah periode sama dengan jumlah baris pada test_df
future = prophet_model.make_future_dataframe(periods=len(test_df), freq='D')
forecast = prophet_model.predict(future)

# Ekstrak prediksi untuk periode test (ambil tail forecast)
prophet_preds = forecast.tail(len(test_df))['yhat'].values

# Evaluasi model Prophet
prophet_mae = mean_absolute_error(test_df['y'], prophet_preds)
prophet_rmse = np.sqrt(mean_squared_error(test_df['y'], prophet_preds))
prophet_accuracy = 100 - (prophet_mae / test_df['y'].mean() * 100)
print(f"Prophet Model Results - MAE: {prophet_mae:.4f}, RMSE: {prophet_rmse:.4f}, Accuracy: {prophet_accuracy:.2f}%")

# Plot komponen forecast (trend, musiman mingguan, dan musiman tahunan)
fig = prophet_model.plot_components(forecast)
plt.show()

# Plot perbandingan nilai aktual vs prediksi
plt.figure(figsize=(14, 7))
plt.plot(test_df['ds'], test_df['y'], label='Actual Price', color='blue')
plt.plot(test_df['ds'], prophet_preds, label='Prophet Prediction', color='red', linestyle='--')
plt.title('Prophet: Actual vs Predicted Cardano Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

"""
### XGBoost Model
"""

X_train = train_data[['priceOpen']]
y_train = train_data['priceClose']
X_test = test_data[['priceOpen']]
y_test = test_data['priceClose']

# Buat dan fit model XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Forecast pada test set
xgb_preds = xgb_model.predict(X_test)

# Evaluasi model XGBoost
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_accuracy = 100 - (xgb_mae / y_test.mean() * 100)
print(f"XGBoost Model Results - MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}, Accuracy: {xgb_accuracy:.2f}%")

# Plot perbandingan nilai aktual vs prediksi
plt.figure(figsize=(10, 6))
plt.plot(test_data['dateOpen'], y_test, label='Actual', color='blue')
plt.plot(test_data['dateOpen'], xgb_preds, label='Predicted', color='red', linestyle='--')
plt.title("XGBoost: Actual vs Predicted Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

"""
### LSTM Model
"""

# Lakukan seed setting untuk konsistensi
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Configure TensorFlow for deterministic operations
    tf.config.experimental.enable_op_determinism()
    
set_seeds(42)

# Data preparation
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Prepare data for LSTM
sequence_length = 60  # Use 60 days of historical data to predict the next day

# Scale data - fit scaler hanya pada training data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data['priceClose'].values.reshape(-1, 1))
test_scaled = scaler.transform(test_data['priceClose'].values.reshape(-1, 1))

# Create sequences untuk training data
X_train, y_train = create_sequences(train_scaled, sequence_length)

# Create sequences untuk test data
X_test, y_test = create_sequences(test_scaled, sequence_length)

print(f"LSTM Training samples: {len(X_train)}")
print(f"LSTM Testing samples: {len(X_test)}")

# Buat LSTM Model
def create_lstm_model(sequence_length):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    # Use consistent optimizer configuration
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Create model
lstm_model = create_lstm_model(sequence_length)

# Train LSTM model with early stopping
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

print("\nTraining LSTM Model...")
history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1,
    shuffle=False  # For consistency
)

# Make predictions
print("\nMaking predictions...")
lstm_preds = lstm_model.predict(X_test, verbose=0)

# Inverse transform predictions and actual values
lstm_preds_scaled = scaler.inverse_transform(lstm_preds)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate LSTM model
lstm_mae = mean_absolute_error(y_test_inv, lstm_preds_scaled)
lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_preds_scaled))
lstm_accuracy = 100 - (lstm_mae / np.mean(y_test_inv) * 100)

# Display results
print("\n" + "="*50)
print("LSTM MODEL EVALUATION RESULTS")
print("="*50)
print(f"MAE (Mean Absolute Error): {lstm_mae:.4f} USD")
print(f"RMSE (Root Mean Square Error): {lstm_rmse:.4f} USD")
print(f"Accuracy: {lstm_accuracy:.2f}%")
print(f"Mean Actual Price: {np.mean(y_test_inv):.4f} USD")
print("="*50)

# Plot actual vs predicted values
test_dates = df['dateOpen'].iloc[train_size + sequence_length:train_size + sequence_length + len(y_test)]
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test_inv, label='Actual Price')
plt.plot(test_dates, lstm_preds_scaled, label='LSTM Prediction', linestyle='--')
plt.title('LSTM: Actual vs Predicted Cardano Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

"""
### Conclusion

Kesimpulannya, model berbasis machine learning (XGBoost) dan deep learning (LSTM) jauh lebih efektif untuk memprediksi harga aset kripto yang memiliki volatilitas tinggi dibandingkan dengan model statistik tradisional (Prophet).
"""
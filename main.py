import requests
import joblib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import numpy as np
import os
import warnings
import time
import random

# --- Konfiguration ---
symbol = "BTC-USD"
model_path = "model.joblib"
scaler_path = "scaler.joblib"
threshold = 0.64
save_dir = os.getcwd()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S UTC')}] {msg}")

def get_data_from_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "1"
    }
    headers = {
        "x-cg-pro-api-key": os.getenv("COINGECKO_API_KEY", "")
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            log(f"❌ Fehler von Coingecko (Status {response.status_code}): {response.text}")
            return pd.DataFrame()
        data = response.json()
        if "prices" not in data:
            log("❌ 'prices' fehlt in der Antwort. Inhalt:")
            log(str(data))
            return pd.DataFrame()
    except Exception as e:
        log(f"❌ Ausnahme bei Coingecko-Abruf: {e}")
        return pd.DataFrame()

    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # ⏱️ Aggregiere zu 5-Minuten-Candles
    df_ohlc = df["price"].resample("5T").ohlc()
    df_ohlc.dropna(inplace=True)
    df_ohlc.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }, inplace=True)

    return df_ohlc

def prepare_features(df):
    df["returns"] = df["Close"].pct_change()
    df["MA"] = df["Close"].rolling(5).mean()
    df["STD"] = df["Close"].rolling(5).std()
    df["HIST"] = df["returns"] - df["returns"].rolling(5).mean()
    df["RSI"] = 50.0  # Dummy
    return df.dropna()

def plot_and_save(df, prediction_label, confidence, candle_time):
    df_plot = df.tail(10).copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    df_plot['date_num'] = mdates.date2num(df_plot.index.to_pydatetime())
    ohlc = df_plot[['date_num', 'Open', 'High', 'Low', 'Close']].values

    fig, ax = plt.subplots(figsize=(12,6))
    width = 0.0008 * 5
    locator = mdates.MinuteLocator(interval=5)
    date_fmt = '%H:%M'

    for d in ohlc:
        date, open_, high, low, close = d
        ax.plot([date, date], [low, high], color='black', linewidth=2)

    candlestick_ohlc(ax, ohlc, width=width, colorup='green', colordown='red', alpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.xaxis.set_major_locator(locator)
    plt.xticks(rotation=45)
    ax.set_title(f"{symbol} - Prognose: {prediction_label} (Confidence: {confidence:.2f})")
    ax.set_xlabel("Zeit (UTC)")
    ax.set_ylabel("Preis")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    confidence_str = f"{confidence:.2f}".replace('.', '_')
    filename = f"{candle_time.strftime('%Y-%m-%d_%H-%M')}_{prediction_label}_{confidence_str}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    plt.close()

    if os.path.exists(full_path):
        log(f"✅ Bild gespeichert: {full_path}")
        try:
            upload_url = "https://colab0815.pythonanywhere.com/upload"
            with open(full_path, 'rb') as f:
                files = {'file': (filename, f, 'image/png')}
                response = requests.post(upload_url, files=files)
                if response.status_code == 200:
                    log(f"🚀 Hochgeladen: {filename}")
                else:
                    log(f"❌ Upload-Fehler: {response.status_code} – {response.text}")
        except Exception as e:
            log(f"❌ Fehler beim Upload: {e}")
    else:
        log(f"❌ Datei nicht gespeichert: {full_path}")

def run_prediction():
    delay = random.uniform(2, 4)
    log(f"⏳ Warte {delay:.2f} Sekunden ...")
    time.sleep(delay)

    df = get_data_from_coingecko()
    if df.empty:
        log("❌ Coingecko-Daten leer oder ungültig.")
        return

    df = prepare_features(df)
    if len(df) < 2:
        log("❌ Nicht genügend Daten.")
        return

    row = df.iloc[-2]
    candle_time = row.name.to_pydatetime()
    features = row[["returns", "MA", "STD", "HIST", "RSI"]].values.reshape(1, -1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    scaled = scaler.transform(features)
    probs = model.predict_proba(scaled)
    prediction = 1 if probs[0, 1] >= threshold else 0
    label = "LONG" if prediction == 1 else "NO_LONG"
    confidence = probs[0, 1]

    log(f"{candle_time} | Prediction: {label} | Confidence: {confidence:.4f}")
    plot_and_save(df, label, confidence, candle_time)

if __name__ == "__main__":
    run_prediction()

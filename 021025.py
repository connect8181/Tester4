Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.15.0-157-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

 System information as of Tue Oct  7 21:09:26 UTC 2025

  System load:  0.02              Processes:             96
  Usage of /:   49.3% of 9.51GB   Users logged in:       0
  Memory usage: 68%               IPv4 address for ens6: 217.154.121.241
  Swap usage:   0%

 * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s
   just raised the bar for easy, resilient and secure K8s cluster deployment.

   https://ubuntu.com/engage/secure-kubernetes-at-the-edge

Expanded Security Maintenance for Applications is not enabled.

7 updates can be applied immediately.
To see these additional updates run: apt list --upgradable

Enable ESM Apps to receive additional future security updates.
See https://ubuntu.com/esm or run: sudo pro status


Last login: Mon Oct  6 22:50:57 2025 from 178.115.67.118
alex@ubuntu:~$ c 38
alex@ubuntu:~/Scripts/Tester38$ cat main7_1.py

import importlib
import subprocess
import sys
import os
import json
import time
import threading
import logging
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
from copy import deepcopy

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

EMAIL_ADDRESS = "connect18181g@gmail.com"
EMAIL_PASSWORD = "xdvo ethw pepz aqxl"
EMAIL_RECEIVER = "connect18181g@gmail.com"

def send_trade_email(trade_info):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"Live Trade Notification: {trade_info.get('exit_reason', trade_info.get('status',''))}"
        body = f"""
Trade Number: {trade_info.get('trade_number', '')}
Type: {trade_info.get('status', '')}
Symbol: {trade_info.get('symbol', '')} | Interval: {trade_info.get('interval', '')}
Entry Price: {trade_info.get('entry_price', 0):.2f}
Exit Price: {trade_info.get('exit_price', 0):.2f}
Position Size: {trade_info.get('position_size', 0):.6f}
P&L: {trade_info.get('pnl', 0):.2f}
Capital After Trade: {trade_info.get('capital_after_entry', 0):.2f}
Timestamp: {trade_info.get('closed_timestamp', trade_info.get('entry_timestamp', ''))}
Model Type: {trade_info.get('model_type', '')}
Stop Loss: {trade_info.get('stop_loss', 0):.2f}
Take Profit: {trade_info.get('take_profit', 0):.2f}
Highest Price Since Entry: {trade_info.get('highest_price', 0):.2f}
        """
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"❌ Fehler beim Senden der E-Mail: {e}")


# --- Installation and Imports ---
def install_if_missing(package_name, pip_name=None):
    try:
        importlib.import_module(package_name)    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package_name])

packages = [
    ('yfinance', None),
    ('pandas', None),
    ('numpy', None),
    ('sklearn', 'scikit-learn'),
    ('flask', None),
    ('binance', 'python-binance'),
    ('imbalanced_learn', 'imbalanced-learn'),    ('requests', None)
]

for pkg, pip_name in packages:
    install_if_missing(pkg, pip_name)

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    RandomOverSampler = None
from flask import Flask, render_template, jsonify, request, send_file
from binance.client import Client
import requests
import json
import joblib
import zipfile
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration and Settings ---
class TradingConfig:
    """Trading configuration and parameters"""
    DEFAULT_PARAMS = {
        'SYMBOL': 'BTCUSDT',
        'INTERVAL': '5m',
        'TRAINING_PERIOD': '60d',
        'BACKTEST_PERIOD': '30d',
        'START_CAPITAL': 10000,
        'MAKER_FEE_RATE': 0.001,
        'TAKER_FEE_RATE': 0.001,
        'STOP_LOSS_PCT': 0.05,
        'TAKE_PROFIT_PCT': 0.008,
        'TRAIL_STOP_PCT': 0.02,
        'BUY_PROB_THRESHOLD': 0.9,
        'MODEL_TYPE': 'HistGradientBoosting',        'CONFIDENCE_THRESHOLD': 0.7,
        'FUTURE_WINDOW': 24,
        'USE_OVERSAMPLING': True,
        'TEST_SIZE': 0.3,
        'SIGNAL_EXIT_THRESHOLD': 0.25,
        'SIGNAL_EXIT_CONSECUTIVE': 3,
        'MIN_HOLD_CANDLES': 6
    }

class SettingsManager:
    """Manage saving and loading of trading parameters"""

    def __init__(self, settings_dir: str = "settings"):
        self.settings_dir = settings_dir
        os.makedirs(settings_dir, exist_ok=True)

    def save_settings(self, params: Dict, name: str = None) -> str:
        """Save parameters to a JSON file"""
        try:
            if name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"settings_{timestamp}"

            if not name.endswith('.json'):
                name += '.json'

            filepath = os.path.join(self.settings_dir, name)

            settings_data = deepcopy(params)
            settings_data['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'name': name,
                'version': '1.0'
            }

            with open(filepath, 'w') as f:
                json.dump(settings_data, f, indent=2)

            logger.info(f"Settings saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            raise e

    def load_settings(self, filepath: str) -> Dict:
        """Load parameters from a JSON file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Settings file not found: {filepath}")

            with open(filepath, 'r') as f:
                settings_data = json.load(f)

            if '_metadata' in settings_data:
                del settings_data['_metadata']

            logger.info(f"Settings loaded from: {filepath}")
            return settings_data

        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            raise e

    def list_saved_settings(self) -> List[Dict]:
        """List all saved settings with metadata"""
        try:
            settings = []
            if not os.path.exists(self.settings_dir):
                return settings

            for filename in os.listdir(self.settings_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.settings_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)

                        metadata = data.get('_metadata', {})
                        settings.append({
                            'filename': filename,
                            'filepath': filepath,
                            'name': metadata.get('name', filename),
                            'saved_at': metadata.get('saved_at', 'Unknown'),
                            'symbol': data.get('SYMBOL', 'Unknown'),
                            'model_type': data.get('MODEL_TYPE', 'Unknown'),
                            'start_capital': data.get('START_CAPITAL', 0)
                        })
                    except Exception as e:
                        logger.error(f"Error reading settings file {filename}: {e}")
                        continue

            settings.sort(key=lambda x: x['saved_at'], reverse=True)
            return settings

        except Exception as e:
            logger.error(f"Error listing settings: {e}")
            return []

    def delete_settings(self, filepath: str) -> bool:
        """Delete a settings file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Settings deleted: {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting settings: {e}")
            return False

# --- Data Provision ---
class DataProvider:
    """Handles data fetching from various sources"""

    @staticmethod
    def get_binance_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Binance API"""
        try:
            client = Client()
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }

            binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_5MINUTE)
            klines = client.get_historical_klines(symbol, binance_interval, start_date, end_date)

            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)

            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        except Exception as e:
            logger.error(f"Binance API error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_yfinance_data(symbol: str, period: str = '60d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            yf_symbol = 'BTC-USD' if symbol == 'BTCUSDT' else symbol
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            if df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in required_cols if col in df.columns]
            if len(available_cols) == len(required_cols):
                df = df[required_cols]
            else:
                logger.error(f"Missing columns in yfinance data. Available: {df.columns.tolist()}")
                return pd.DataFrame()

            return df
        except Exception as e:
            logger.error(f"YFinance error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_training_data(symbol: str, period: str = '60d', interval: str = '5m') -> pd.DataFrame:
        """Get data specifically for model training"""
        logger.info(f"🔧 TRAINING DATA: Using Yahoo Finance")
        return DataProvider.get_yfinance_data(symbol, period, interval)

    @staticmethod
    def get_backtest_data_with_period(symbol: str, period: str = '30d', interval: str = '5m') -> pd.DataFrame:
        """Get data specifically for backtesting with configurable period"""
        data = DataProvider.get_binance_data(symbol, interval,
            (datetime.now() - timedelta(days=int(period.rstrip('d')))).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d'))
        if data.empty:
            yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
            data = DataProvider.get_yfinance_data(yf_symbol, period, interval)
        return data

    @staticmethod
    def get_live_data(symbol: str, interval: str = '5m') -> pd.DataFrame:
        """
        Get live data for live trading - uses same source as training.
        Fetches the last 5 days (hardcoded in yfinance) to ensure enough
        history for feature extraction (like RSI 14).
        """
        logger.info(f"📡 LIVE TRADING: Getting latest data (5d history)")
        try:
            yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
            # Hardcoded '5d' period to ensure enough lookback data for TA features
            df = DataProvider.get_yfinance_data(yf_symbol, '5d', interval)
            if not df.empty:
                logger.info(f"✅ Live data: {len(df)} candles from Yahoo Finance")
                return df
            else:
                raise Exception("Empty dataframe from Yahoo Finance")
        except Exception as yf_error:
            logger.error(f"❌ Yahoo Finance error: {yf_error}")
        return pd.DataFrame()

# --- Technical Analysis ---
class TechnicalAnalysis:
    """Technical analysis and feature extraction"""
    @staticmethod
    def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return pd.Series([50] * len(series), index=series.index)

    @staticmethod
    def extract_features(df: pd.DataFrame, future_window: int = 24, is_live_trading: bool = False) -> pd.DataFrame:
        """Extract technical features using simplified approach"""
        try:
            # Increased minimum length to ensure RSI (window=14) can be calculated
            MIN_DATA_LENGTH = 20

            if df.empty or len(df) < MIN_DATA_LENGTH:
                if is_live_trading:
                    logger.warning(f"Insufficient data for feature extraction (Min {MIN_DATA_LENGTH}, Got {len(df)})")
                return pd.DataFrame()

            df = df.copy()

            if is_live_trading:
                logger.info(f"🔧 LIVE FEATURE EXTRACTION:")
                logger.info(f"   Input data shape: {df.shape}")
                logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
                logger.info(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=5).std()
            df['rsi'] = TechnicalAnalysis.compute_rsi(df['Close'])

            # This is the critical line that removes rows with NaN features (e.g., the first 14 rows due to RSI)
            df = df.dropna()

            if df.empty:
                logger.error("All data was NaN after feature extraction")
                return pd.DataFrame()

            # Ensure at least 1 candle with features remains for live prediction
            # --- FIX: SYNTAX ERROR WAS HERE ---
            if is_live_trading and len(df) < 1:
                logger.error("❌ Not enough data points remain after feature extraction.")                return pd.DataFrame()
            # -----------------------------------


            if is_live_trading:
                logger.info(f"   ✅ Final features shape: {df.shape}")
                latest = df.iloc[-1]
                logger.info(f"   ✅ Latest: returns={latest['returns']:.6f}, volatility={latest['volatility']:.6f}, rsi={latest['rsi']:.2f}")

            return df

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            traceback.print_exc()
            return pd.DataFrame()

# --- ML Model ---
class MLModel:
    """Enhanced machine learning model for trading predictions"""

    def __init__(self, model_type: str = 'HistGradientBoosting'):
        self.model_type = model_type
        self.model = self._get_model(model_type)
        self.scaler = StandardScaler()
        self.features = ['returns', 'volatility', 'rsi']
        self.is_trained = False
        self.training_accuracy = 0.0
        self.test_accuracy = 0.0
        self.validation_accuracy = 0.0
        self.hit_rate = 0.0
        self.class_distribution = {}

    def _get_model(self, model_type: str):
        """Get the appropriate model based on type"""
        if model_type == 'HistGradientBoosting':
            return HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            logger.warning(f"Unknown model type {model_type}, using HistGradientBoosting")            return HistGradientBoostingClassifier(max_iter=200, random_state=42)

    def train(self, df: pd.DataFrame, future_window: int = 24, take_profit_pct: float = 0.008,
              use_oversampling: bool = True, test_size: float = 0.3, training_period: str = '60d'):
        """
        Training method for the model.
        """
        try:
            if df.empty or len(df) < 200:
                logger.error(f"Insufficient data for training: {len(df)} rows")
                return 0.0

            df = df.copy()

            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            df.dropna(inplace=True)

            target_dist = df['target'].value_counts(normalize=True)
            self.class_distribution = target_dist.to_dict()
            logger.info(f"Target distribution: {self.class_distribution}")

            X = df[self.features]
            y = df['target']

            if len(y.unique()) < 2:
                logger.error("Target variable has only one class")
                return 0.0

            X_scaled = self.scaler.fit_transform(X)

            if use_oversampling and len(y.unique()) == 2 and RandomOverSampler is not None:
                try:
                    ros = RandomOverSampler(random_state=42)
                    X_res, y_res = ros.fit_resample(X_scaled, y)
                    logger.info(f"Applied oversampling: {len(X_scaled)} -> {len(X_res)} samples")
                except Exception as e:
                    logger.warning(f"Oversampling failed, using original data: {e}")
                    X_res, y_res = X_scaled, y
            else:
                if RandomOverSampler is None:                    logger.warning("RandomOverSampler not available, using original data")                X_res, y_res = X_scaled, y

            logger.info(f"Training {self.model_type} model...")
            self.model.fit(X_res, y_res)
            self.is_trained = True

            self.training_accuracy = self.model.score(X_scaled, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, shuffle=False, random_state=42
            )

            self.test_accuracy = self.model.score(X_test, y_test)
            self.validation_accuracy = self.test_accuracy
            self.hit_rate = self.test_accuracy

            logger.info(f"=== Training Results ===")
            logger.info(f"📊 Model trained with Accuracy: {self.training_accuracy:.4f}")
            logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
            logger.info(f"Hit Rate: {self.hit_rate:.4f}")
            logger.info(f"Model Type: {self.model_type}")
            logger.info(f"Features Used: {len(self.features)} - {self.features}")

            return self.training_accuracy
        except Exception as e:
            logger.error(f"Model training error: {e}")
            traceback.print_exc()
            return 0.0

    def predict(self, features: np.ndarray, is_live_trading: bool = False) -> Tuple[float, float]:
        """Make prediction and return probability and confidence"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained or None")
                return 0.5, 0.5

            if len(features) != len(self.features):
                logger.error(f"Feature mismatch: expected {len(self.features)}, got {len(features)}")
                return 0.5, 0.5

            if is_live_trading:
                logger.info(f"🔍 LIVE PREDICTION:")
                logger.info(f"   Features: {[f'{val:.6f}' for val in features]}")

                nan_check = [np.isnan(val) or np.isinf(val) for val in features]
                if any(nan_check):
                    logger.error(f"     INVALID FEATURES: {nan_check}")
                    return 0.5, 0.5

            X_scaled = self.scaler.transform([features])

            if is_live_trading:
                logger.info(f"   Scaled: {[f'{val:.6f}' for val in X_scaled[0]]}")

            probabilities = self.model.predict_proba(X_scaled)[0]

            if is_live_trading:
                logger.info(f"   Raw probabilities: {[f'{p:.6f}' for p in probabilities]}")

            if len(probabilities) > 1:
                probability = probabilities[1]
                confidence = max(probabilities)
            else:
                probability = 0.5
                confidence = 0.5

            if is_live_trading:
                logger.info(f"   ✅ Buy prob: {probability:.6f}, Confidence: {confidence:.6f}")

            return float(probability), float(confidence)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5, 0.5

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        try:
            if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
                return {}

            importance_dict = dict(zip(self.features, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return {}

    def save_model(self, filepath: str = None) -> str:
        """Save model and scaler to file"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")

            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"models/trading_model_{self.model_type}_{timestamp}.pkl"

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'features': self.features,
                'training_accuracy': self.training_accuracy,
                'test_accuracy': self.test_accuracy,
                'validation_accuracy': self.validation_accuracy,
                'hit_rate': self.hit_rate,
                'class_distribution': self.class_distribution,
                'saved_at': datetime.now().isoformat()
            }

            joblib.dump(model_data, filepath)            logger.info(f"Model saved successfully to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise e

    def load_model(self, filepath: str) -> bool:
        """Load model and scaler from file"""        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")

            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.features = model_data['features']
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.test_accuracy = model_data.get('test_accuracy', 0.0)
            self.validation_accuracy = model_data.get('validation_accuracy', 0.0)
            self.hit_rate = model_data.get('hit_rate', 0.0)
            self.class_distribution = model_data.get('class_distribution', {})
            self.is_trained = True

            logger.info(f"Model loaded successfully from: {filepath}")
            logger.info(f"Model type: {self.model_type}, Accuracy: {self.training_accuracy:.4f}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    @staticmethod
    def list_saved_models(models_dir: str = "models") -> List[Dict]:
        """List all saved models with their metadata"""
        try:
            if not os.path.exists(models_dir):
                return []

            models = []
            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl'):                    filepath = os.path.join(models_dir, filename)
                    try:
                        model_data = joblib.load(filepath)
                        models.append({
                            'filename': filename,
                            'filepath': filepath,
                            'model_type': model_data.get('model_type', 'Unknown'),
                            'training_accuracy': model_data.get('training_accuracy', 0.0),                            'saved_at': model_data.get('saved_at', 'Unknown'),
                            'features': model_data.get('features', [])
                        })
                    except Exception as e:
                        logger.error(f"Error reading model file {filename}: {e}")
                        continue

            models.sort(key=lambda x: x['saved_at'], reverse=True)
            return models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'test_accuracy': self.test_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'hit_rate': self.hit_rate,
            'features': self.features,
            'n_features': len(self.features),            'class_distribution': self.class_distribution,
            'feature_importance': self.get_feature_importance()
        }

class BacktestEngine:
    """Backtesting engine"""

    def __init__(self):
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}        self.logs = []
        self.running = False

    def run(self, params: Dict, start_date: str, end_date: str):
        """Run backtest"""
        self.running = True
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': True}
        self.logs = []

        try:
            self._log(f"🚀 Starting backtest: {params['SYMBOL']} ({start_date} to {end_date})")

            backtest_period = params.get('BACKTEST_PERIOD', '30d')
            data = self._get_backtest_data(params['SYMBOL'], params['INTERVAL'], start_date, end_date, backtest_period)
            if data.empty:
                self._log("❌ No data available")
                return

            self._log(f"✅ Retrieved {len(data)} candles using {backtest_period} period")

            if len(data) > 5000:
                data = data.tail(5000)
                self._log(f"📊 Limited to {len(data)} candles for performance")

            self._log("🔧 Extracting technical features...")
            data = TechnicalAnalysis.extract_features(data, params['FUTURE_WINDOW'])

            if data.empty or len(data) < 200:                self._log("❌ Insufficient data after feature extraction")
                return

            self._log(f"✅ Features extracted from {len(data)} candles")

            model = MLModel(params['MODEL_TYPE'])
            training_period = params.get('TRAINING_PERIOD', '60d')
            accuracy = model.train(
                data,
                params['FUTURE_WINDOW'],
                params['TAKE_PROFIT_PCT'],
                params.get('USE_OVERSAMPLING', True),
                params.get('TEST_SIZE', 0.3),                training_period
            )

            global current_trained_model
            current_trained_model = model
            global latest_model_info
            model_info = model.get_model_info()
            latest_model_info = {
                'training_accuracy': model_info.get('training_accuracy', 0.0),
                'test_accuracy': model_info.get('test_accuracy', 0.0),
                'validation_accuracy': model_info.get('validation_accuracy', 0.0),
                'hit_rate': model_info.get('hit_rate', 0.0),
                'model_type': model_info.get('model_type', 'Unknown'),
                'features_count': model_info.get('n_features', 0),
                'class_distribution': model_info.get('class_distribution', {})
            }
            self._log(f"📊 Model Info: {model_info['model_type']}, Features: {model_info['n_features']}")

            split_idx = int(len(data) * (1 - params.get('TEST_SIZE', 0.3)))
            test_data = data.iloc[split_idx:]            self._log(f"📚 Training: {len(data.iloc[:split_idx])} candles, Testing: {len(test_data)} candles")

            if accuracy == 0.0:
                self._log("❌ Model training failed")
                return

            self._log(f"🤖 Model trained with {accuracy:.2%} accuracy")

            self._simulate_trading(model, test_data, params)

        except Exception as e:
            self._log(f"❌ Backtest error: {str(e)}")
            logger.error(f"Backtest error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self.results['running'] = False

    def _get_backtest_data(self, symbol: str, interval: str, start_date: str, end_date: str, period: str = '30d') -> pd.DataFrame:
        """Get data for backtesting"""
        has_dates = start_date and end_date and start_date.strip() and end_date.strip()

        if has_dates:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                days_diff = (end_dt - start_dt).days
                self._log(f"📅 Using specific date range: {start_date} to {end_date} ({days_diff} days)")

                data = DataProvider.get_binance_data(symbol, interval, start_date, end_date)

                if data.empty:
                    calculated_period = f"{max(days_diff, 7)}d"
                    self._log(f"🔄 Binance unavailable, using Yahoo Finance with {calculated_period}...")
                    yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
                    data = DataProvider.get_yfinance_data(yf_symbol, calculated_period, interval)

                    if not data.empty:
                        try:
                            start_dt_pd = pd.to_datetime(start_date)
                            end_dt_pd = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                            data = data[(data.index >= start_dt_pd) & (data.index < end_dt_pd)]
                            self._log(f"🤔 Filtered to exact date range: {len(data)} candles")
                        except Exception as e:
                            self._log(f"⚠️ Date filtering failed: {e}")
                return data

            except Exception as e:
                self._log(f"📅 Date parsing error: {e}, falling back to period: {period}")

        self._log(f"📅 Using backtest period: {period} (no specific dates)")
        data = DataProvider.get_backtest_data_with_period(symbol, period, interval)

        return data

    def _simulate_trading(self, model: MLModel, data: pd.DataFrame, params: Dict):
        """Simulate trading on historical data"""
        try:
            capital = params['START_CAPITAL']            position = 0
            btc_amount = 0
            entry_price = 0
            trades = []
            equity_curve = []
            total_fees = 0
            open_trade_levels = {}
            highest_price_since_entry = 0
            self._log(f"💰 Starting simulation with ${capital:,.2f}")

            chunk_size = 100
            total_processed = 0

            trade_entry_timestamp = None
            buy_probability = 0.0

            # The stop-loss is triggered on the low of the candle, not the close
            # to prevent it from being "skipped" during a sharp drop.

            for chunk_start in range(0, len(data), chunk_size):
                if not self.running:
                    break

                chunk_end = min(chunk_start + chunk_size, len(data))
                chunk_data = data.iloc[chunk_start:chunk_end]

                for i, (timestamp, candle) in enumerate(chunk_data.iterrows()):
                    if not self.running:
                        break

                    try:
                        price = float(candle['Close'])
                        equity = capital if position == 0 else (btc_amount * price)
                        equity_curve.append({'timestamp': str(timestamp), 'equity': equity})

                        try:
                            available_features = [f for f in model.features if f in candle.index]
                            if len(available_features) != len(model.features):
                                logger.warning(f"Missing features: {set(model.features) - set(available_features)}")
                                continue
                            features = candle[model.features].values
                            prob, confidence = model.predict(features)
                        except Exception as e:
                            logger.error(f"Feature extraction error for prediction: {e}")
                            continue

                        # Buy signal
                        if (position == 0 and                            prob > params['BUY_PROB_THRESHOLD'] and
                            confidence > params['CONFIDENCE_THRESHOLD']):

                            buy_fee = capital * params['TAKER_FEE_RATE']
                            btc_amount = (capital - buy_fee) / price
                            entry_price = price
                            highest_price_since_entry = price
                            position = 1
                            take_profit = entry_price * (1 + params['TAKE_PROFIT_PCT'])
                            stop_loss = entry_price * (1 - params['STOP_LOSS_PCT'])
                            open_trade_levels = {
                                'take_profit': take_profit,
                                'stop_loss': stop_loss,
                                'buy_fee': buy_fee,
                                'entry_confidence': confidence
                            }
                            self._candles_in_position = 0
                            self._low_prob_count = 0

                            trade_entry_timestamp = timestamp
                            buy_probability = prob

                        # Sell signal
                        elif position == 1:
                            if price > highest_price_since_entry:
                                highest_price_since_entry = price

                            stop_loss = entry_price * (1 - params['STOP_LOSS_PCT'])
                            take_profit = entry_price * (1 + params['TAKE_PROFIT_PCT'])
                            trailing_stop = highest_price_since_entry * (1 - params.get('TRAIL_STOP_PCT', 0.02))

                            candles_in_position = getattr(self, '_candles_in_position', 0) + 1
                            self._candles_in_position = candles_in_position

                            if not hasattr(self, '_low_prob_count'):
                                self._low_prob_count = 0

                            if prob < 0.3:
                                self._low_prob_count += 1
                            else:
                                self._low_prob_count = 0

                            exit_reason = None

                            # Check low price for stop-loss and high price for take-profit                            if float(candle['Low']) <= stop_loss:
                                exit_reason = 'Stop Loss'
                                exit_price = stop_loss
                            elif float(candle['High']) >= take_profit:
                                exit_reason = 'Take Profit'
                                exit_price = take_profit
                            elif price <= trailing_stop and price >= entry_price * 1.001:
                                exit_reason = 'Trailing Stop'
                                exit_price = price
                            elif (self._low_prob_count >= 3 and
                                  candles_in_position > 6 and
                                  prob < 0.25):
                                exit_reason = 'Signal Exit'
                                exit_price = price

                            if exit_reason:
                                sell_fee = btc_amount * exit_price * params['TAKER_FEE_RATE']
                                total_trade_fees = open_trade_levels['buy_fee'] + sell_fee                                total_fees += total_trade_fees

                                raw_pnl = (exit_price - entry_price) * btc_amount
                                pnl = raw_pnl - total_trade_fees

                                capital += pnl + total_trade_fees

                                duration = timestamp - trade_entry_timestamp if trade_entry_timestamp else pd.Timedelta(0)

                                trades.append({
                                    'timestamp': str(timestamp),
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'exit_reason': exit_reason,
                                    'pnl': pnl,
                                    'raw_pnl': raw_pnl,
                                    'buy_probability': buy_probability,
                                    'confidence': confidence,
                                    'entry_confidence': open_trade_levels.get('entry_confidence', 0),
                                    'fees': total_trade_fees,
                                    'highest_price': highest_price_since_entry,
                                    'hold_duration': candles_in_position,
                                    'entry_date': str(trade_entry_timestamp) if trade_entry_timestamp else str(timestamp),
                                    'exit_date': str(timestamp),
                                    'status': 'CLOSED',
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'position_size': btc_amount,
                                    'trade_usd_amount': params['START_CAPITAL'] - open_trade_levels['buy_fee'],
                                    'duration': str(duration)
                                })

                                trade_type = "WIN" if pnl > 0 else "LOSS"
                                self._log(f" ⭐ {trade_type}: ${entry_price:.2f} → ${exit_price:.2f} | P&L: ${raw_pnl:.2f} | P&L (net): ${pnl:.2f} | Fees: ${total_trade_fees:.2f}")                                position = 0
                                self._candles_in_position = 0
                                self._low_prob_count = 0
                                open_trade_levels = {}
                                self.results['trades'] = trades
                                trade_entry_timestamp = None
                                buy_probability = 0.0

                        total_processed += 1

                    except Exception as e:
                        logger.error(f"Error processing candle {i}: {e}")
                        continue

                progress = (chunk_end / len(data)) * 100
                self._log(f"📈 Processing... {progress:.1f}% complete ({len(trades)} trades so far)")

            if trades:
                self._calculate_metrics(trades, params['START_CAPITAL'], total_fees)
            else:
                self._log("⚠️ No trades executed during backtest")
                self.results['metrics'] = {
                    'total_trades': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_win': 0,
                    'avg_loss': 0, 'return_pct': 0, 'total_fees': 0
                }

            final_capital = params['START_CAPITAL'] + sum(t['pnl'] for t in trades)
            self._log(f"💸 Final Capital: ${final_capital:.2f}")

            self.results['trades'] = trades
            self.results['equity_curve'] = equity_curve

            self._log(f"✅ Backtest completed - {len(trades)} trades executed")

        except Exception as e:
            self._log(f"❌ Backtest error: {str(e)}")
            logger.error(f"Backtest error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self.results['running'] = False

    def _get_backtest_data(self, symbol: str, interval: str, start_date: str, end_date: str, period: str = '30d') -> pd.DataFrame:
        """Get data for backtesting"""
        has_dates = start_date and end_date and start_date.strip() and end_date.strip()

        if has_dates:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                days_diff = (end_dt - start_dt).days
                self._log(f"📅 Using specific date range: {start_date} to {end_date} ({days_diff} days)")

                data = DataProvider.get_binance_data(symbol, interval, start_date, end_date)

                if data.empty:
                    calculated_period = f"{max(days_diff, 7)}d"
                    self._log(f"🔄 Binance unavailable, using Yahoo Finance with {calculated_period}...")
                    yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
                    data = DataProvider.get_yfinance_data(yf_symbol, calculated_period, interval)

                    if not data.empty:
                        try:
                            start_dt_pd = pd.to_datetime(start_date)
                            end_dt_pd = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                            data = data[(data.index >= start_dt_pd) & (data.index < end_dt_pd)]
                            self._log(f"🤔 Filtered to exact date range: {len(data)} candles")
                        except Exception as e:
                            self._log(f"⚠️ Date filtering failed: {e}")
                return data

            except Exception as e:
                self._log(f"📅 Date parsing error: {e}, falling back to period: {period}")

        self._log(f"📅 Using backtest period: {period} (no specific dates)")
        data = DataProvider.get_backtest_data_with_period(symbol, period, interval)

        return data

    def _calculate_metrics(self, trades: List[Dict], start_capital: float, total_fees: float):
        """Calculate backtest performance metrics"""
        try:
            if not trades:
                self.results['metrics'] = {}
                return

            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]

            winner_amount = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            loser_amount = sum([t['pnl'] for t in losing_trades]) if losing_trades else 0
            total_pnl_with_fees = winner_amount + loser_amount

            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = winner_amount / len(winning_trades) if winning_trades else 0
            avg_loss = abs(loser_amount) / len(losing_trades) if losing_trades else 0
            win_loss_ratio = len(winning_trades) / len(losing_trades) if losing_trades else len(winning_trades)

            return_pct = (total_pnl_with_fees / start_capital) * 100

            metrics = {
                'total_trades': len(trades),
                'winner_count': len(winning_trades),
                'winner_amount': winner_amount,
                'loser_count': len(losing_trades),
                'loser_amount': abs(loser_amount),
                'total_pnl': total_pnl_with_fees,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'return_pct': return_pct,
                'total_fees': total_fees
            }

            self.results['metrics'] = metrics            self._log(f"💰 === P&L INCLUDING FEES ===")
            self._log(f"💰 Total P&L (with fees): ${total_pnl_with_fees:.2f}")
            self._log(f"💰 Total Fees Paid: ${total_fees:.2f}")
            self._log(f"📊 Final Results: {len(trades)} trades, {len(winning_trades)} winners (${winner_amount:.2f}), {len(losing_trades)} losers (${abs(loser_amount):.2f})")
            self._log(f"📊 Win/Loss Ratio: {win_loss_ratio:.2f}")
            self._log(f"📊 Total Return: {return_pct:.2f}%")

        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            self.results['metrics'] = {}

    def _log(self, message: str):
        """Add log message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} | {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
        logger.info(message)

    def stop(self):
        """Stop backtest"""
        self.running = False
        self._log("🛑 Backtest stopped by user")

class LiveTrader:
    """Live trader with single active trade and detailed logging, matching backtester logic."""

    def __init__(self):
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}        self.logs = []
        self.running = False
        self.model = None
        self.params = {}
        self.data_history = pd.DataFrame()
        self._last_candle_timestamp = None

    def start(self, params, model):
        full_params = deepcopy(TradingConfig.DEFAULT_PARAMS)
        full_params.update(params)
        self.params = full_params
        self.running = True
        self.model = model
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': True}
        self.logs = []
        self._last_candle_timestamp = None

        try:
            script_filepath = os.path.abspath(__file__)
            self._log(f"✅ Script Filepath: {script_filepath}")
        except NameError:
            self._log("⚠️ Could not determine script path.")

        self._log(f"🚀 Live trading started with ${self.params.get('START_CAPITAL', 10000):.2f}")
        self._log(f"📊 Model: {model.model_type}, Accuracy: {getattr(model, 'training_accuracy', 0.0):.2%}")
        self._log(f"📍 Symbol: {self.params.get('SYMBOL', 'Unknown')} | Interval: {self.params.get('INTERVAL', 'Unknown')}")
        self._log(self._get_settings_log())

        # Initial data fetch and feature extraction
        initial_data = DataProvider.get_live_data(self.params['SYMBOL'], self.params['INTERVAL'])

        # NOTE: Using the entire initial data set for feature extraction
        self.data_history = TechnicalAnalysis.extract_features(
            initial_data, self.params.get('FUTURE_WINDOW', 1), is_live_trading=True
        )

        if self.data_history.empty:
            self._log("❌ Initial data fetch failed or insufficient data after feature extraction. Cannot start live trading.")
            self.running = False
            return

        self._last_candle_timestamp = self.data_history.index[-1]
        self._log(f"✅ Initial data history loaded up to: {self._last_candle_timestamp}")

        threading.Thread(target=self._live_trading_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self._log("🛑 Live trading stopped")

    def _get_and_process_data(self) -> pd.DataFrame:
        """Holt die neuesten Daten und extrahiert Features."""
        data = DataProvider.get_live_data(self.params['SYMBOL'], self.params['INTERVAL'])
        if data.empty or data.index.empty:
            return pd.DataFrame()

        featured_data = TechnicalAnalysis.extract_features(
            data, self.params.get('FUTURE_WINDOW', 1), is_live_trading=True
        )
        return featured_data


    # --- FIX: CORRECTED LIVE TRADING LOOP FOR DELAY ISSUE ---
    def _live_trading_loop(self):
        try:
            interval_str = self.params.get('INTERVAL', '5m')
            # Extract interval in minutes (works for '5m', '15m', '1h', etc.)
            if 'm' in interval_str:
                interval_minutes = int(interval_str.rstrip('m'))
            elif 'h' in interval_str:
                interval_minutes = int(interval_str.rstrip('h')) * 60
            else:
                interval_minutes = 5 # Default to 5 minutes

            while self.running:
                # 1. Calculate time until the next expected candle CLOSE time
                now = datetime.now()
                # Find the next minute that is a multiple of the interval_minutes past the hour
                current_minute = now.minute

                # Minutes until the next interval boundary
                minutes_to_wait = (interval_minutes - (current_minute % interval_minutes)) % interval_minutes

                # Seconds remaining in the current minute (plus the full minutes to wait)
                seconds_to_wait = (minutes_to_wait * 60) + (60 - now.second)

                # We wait until *after* the candle close, plus a small API buffer (3s)
                wait_time_with_buffer = seconds_to_wait + 3

                # If we calculated a very small wait time (meaning we just missed the candle close),
                # we wait for the *next* full interval instead.
                if wait_time_with_buffer < 15:
                    wait_time_with_buffer += interval_minutes * 60

                self._log(f"💤 Waiting {wait_time_with_buffer:.1f}s for the next candle close at approx. { (now + timedelta(seconds=wait_time_with_buffer)).strftime('%H:%M:%S') }...")

                time.sleep(wait_time_with_buffer)

                # 2. Short-Retry-Loop: Poll quickly until the new candle appears
                retries = 0
                max_retries = 10 # 10 retries * 5s = 50s total buffer

                while self.running and retries < max_retries:
                    featured_data = self._get_and_process_data()

                    if featured_data.empty:
                        self._log("❌ Data fetch/feature extraction failed. Retrying in 5s.")
                        time.sleep(5)
                        retries += 1
                        continue

                    # Filter for candles that are truly new (index > last processed timestamp)
                    new_candles = featured_data[featured_data.index > self._last_candle_timestamp]

                    if not new_candles.empty:                        # Process only the very latest candle
                        latest_candle = featured_data.iloc[-1]
                        self._log(f"🔄 Found {len(new_candles)} new candle(s) after {retries} retries.")

                        # Process the latest candle for trading decisions
                        self._process_single_candle(latest_candle)

                        # Update history and last timestamp
                        self._last_candle_timestamp = latest_candle.name
                        self.data_history = featured_data.tail(50)
                        break # Exit retry loop, return to main sleep
                    else:
                        # Candle not yet available (API delay)
                        self._log(f"⚠️ No new candle found since {self._last_candle_timestamp}. Retry {retries+1}/{max_retries} in 5s.")                        retries += 1
                        time.sleep(5)

                if retries >= max_retries:
                    self._log("🚨 MAX RETRIES REACHED. Candle seriously delayed or API down. Resuming initial sleep cycle.")

        except Exception as e:
            self._log(f"❌ Live trading loop error: {e}")
            self.running = False
            traceback.print_exc()
    # ----------------------------------------------------


    def _process_single_candle(self, candle: pd.Series):
        """Processes a single candle for live trading decisions"""
        try:
            price = float(candle['Close'])
            timestamp = candle.name

            capital = self.params.get('START_CAPITAL', 10000)

            trades = self.results.get('trades', [])
            equity_curve = self.results.get('equity_curve', [])
            open_trade = next((t for t in trades if t.get('status') == 'OPEN'), None)

            equity = capital if not open_trade else open_trade.get('capital_after_entry', 0) + open_trade.get('unrealized_pnl', 0)
            equity_curve.append({'timestamp': str(timestamp), 'equity': equity})
            self.results['equity_curve'] = equity_curve

            # Add a check to ensure features are present
            if not all(f in candle.index for f in self.model.features):
                self._log("❌ Missing features in candle. Cannot make a prediction.")
                return

            features = candle[self.model.features].values
            prob, confidence = self.model.predict(features, is_live_trading=True)

            self._log(
                f"🔍 Candle {timestamp} | Price=${price:.2f} | BuyProb={prob:.3f} | Conf={confidence:.3f} | Position: {'Open' if open_trade else 'Cash'}"
            )

            stop_loss_pct = self.params.get('STOP_LOSS_PCT')
            take_profit_pct = self.params.get('TAKE_PROFIT_PCT')
            trail_stop_pct = self.params.get('TRAIL_STOP_PCT')
            signal_exit_threshold = self.params.get('SIGNAL_EXIT_THRESHOLD')
            signal_exit_consecutive = self.params.get('SIGNAL_EXIT_CONSECUTIVE')
            min_hold_candles = self.params.get('MIN_HOLD_CANDLES')
            buy_prob_threshold = self.params.get('BUY_PROB_THRESHOLD')
            confidence_threshold = self.params.get('CONFIDENCE_THRESHOLD')
            taker_fee_rate = self.params.get('TAKER_FEE_RATE')

            # BUY
            if not open_trade and prob > buy_prob_threshold and confidence > confidence_threshold:
                position_size_usd = capital * 0.99
                fee = position_size_usd * taker_fee_rate
                btc_amount = (position_size_usd - fee) / price
                entry_price = price
                highest_price_since_entry = price
                capital_after_entry = capital - position_size_usd

                new_trade = {
                    'trade_number': len(trades) + 1,
                    'entry_timestamp': str(timestamp),
                    'entry_price': entry_price,
                    'position_size': btc_amount,
                    'trade_usd_amount': position_size_usd, # New field for USDT amount
                    'status': 'OPEN',
                    'capital_after_entry': capital_after_entry,
                    'highest_price': highest_price_since_entry,
                    'hold_duration': 1,
                    'low_prob_count': 0,
                    'total_fees': fee,
                    'current_price': price,
                    'unrealized_pnl': 0.0,
                    'stop_loss': entry_price * (1 - stop_loss_pct),
                    'take_profit': entry_price * (1 + take_profit_pct),
                    'signal_exit_threshold': signal_exit_threshold,
                    'signal_exit_consecutive': signal_exit_consecutive,
                    'min_hold_candles': min_hold_candles,
                    'symbol': self.params.get('SYMBOL', ''),
                    'interval': self.params.get('INTERVAL', ''),
                    'model_type': self.model.model_type
                }
                trades.append(new_trade)
                self.results['trades'] = trades
                self._log(f"🚀 Trade #{len(trades)} BUY TRIGGERED: Entry=${entry_price:.2f}")
                send_trade_email(new_trade)

            # ON HOLD / SELL
            elif open_trade:
                highest_price = max(open_trade['highest_price'], price)
                hold_duration = open_trade['hold_duration'] + 1
                low_prob_count = open_trade['low_prob_count'] + 1 if prob < 0.3 else 0

                unrealized_pnl = (price - open_trade['entry_price']) * open_trade['position_size']

                open_trade.update({
                    'current_price': price,
                    'highest_price': highest_price,
                    'hold_duration': hold_duration,
                    'low_prob_count': low_prob_count,
                    'unrealized_pnl': unrealized_pnl,
                })

                stop_loss = open_trade['entry_price'] * (1 - stop_loss_pct)
                take_profit = open_trade['entry_price'] * (1 + take_profit_pct)
                trailing_stop = highest_price * (1 - trail_stop_pct)

                exit_reason = None
                exit_price = price

                # Check low price for stop-loss and high price for take-profit
                if float(candle['Low']) <= stop_loss:
                    exit_reason = 'Stop Loss'                    exit_price = stop_loss
                elif float(candle['High']) >= take_profit:
                    exit_reason = 'Take Profit'
                    exit_price = take_profit
                elif price <= trailing_stop and price >= open_trade['entry_price'] * 1.001:
                    exit_reason = 'Trailing Stop'
                    exit_price = price
                elif (low_prob_count >= signal_exit_consecutive and
                      hold_duration > min_hold_candles and
                      prob < signal_exit_threshold):
                    exit_reason = 'Signal Exit'
                    exit_price = price

                if exit_reason:
                    exit_fee = open_trade['position_size'] * exit_price * taker_fee_rate
                    total_fees = open_trade['total_fees'] + exit_fee

                    raw_pnl = (exit_price - open_trade['entry_price']) * open_trade['position_size']
                    pnl = raw_pnl - total_fees

                    open_trade.update({
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'status': 'CLOSED',
                        'raw_pnl': raw_pnl,
                        'pnl': pnl,
                        'fees': total_fees,
                        'unrealized_pnl': 0.0,
                        'closed_timestamp': str(timestamp)
                    })

                    self._log(f"🎯 Trade #{open_trade['trade_number']} SELL ({exit_reason}): Exit=${exit_price:.2f}, P&L=${pnl:.2f}")
                    send_trade_email(open_trade)

            log_data = {
                'price': f"${price:.2f}",
                'Next TP price': f"${open_trade['take_profit']:.2f}" if open_trade else "N/A",
                'Next SL price': f"${open_trade['stop_loss']:.2f}" if open_trade else "N/A",
                'Next Signal exit prob': f"{signal_exit_threshold:.2%}",
                'Next Trailstop price': f"${trailing_stop:.2f}" if open_trade and 'trailing_stop' in locals() else "N/A",
                'conf': f"{confidence:.2%}",
                'Buy proba': f"{prob:.2%}"
            }
            self._log(f"📊 DATA: {log_data}")

        except Exception as e:
            self._log(f"❌ Live simulation error: {str(e)}")
            traceback.print_exc()

    def manual_close_trade(self) -> Tuple[bool, str]:
        """Manually close an open trade at the current market price."""
        try:
            open_trade = next((t for t in self.results['trades'] if t.get('status') == 'OPEN'), None)

            if not open_trade:
                self._log("MANUAL CLOSE: No open trade to close.")
                return False, "No open trade to close."

            if self.data_history.empty:
                self._log("MANUAL CLOSE: Cannot close, no recent price data available.")
                return False, "No recent price data available."

            exit_price = self.data_history['Close'].iloc[-1]
            timestamp = self.data_history.index[-1]
            taker_fee_rate = self.params.get('TAKER_FEE_RATE')

            exit_fee = open_trade['position_size'] * exit_price * taker_fee_rate
            total_fees = open_trade['total_fees'] + exit_fee
            raw_pnl = (exit_price - open_trade['entry_price']) * open_trade['position_size']
            pnl = raw_pnl - total_fees

            open_trade.update({
                'exit_price': exit_price,
                'exit_reason': 'Manual Close',
                'status': 'CLOSED',
                'raw_pnl': raw_pnl,
                'pnl': pnl,
                'fees': total_fees,
                'unrealized_pnl': 0.0,
                'closed_timestamp': str(timestamp)
            })

            self._log(f"✅ MANUAL CLOSE: Trade #{open_trade['trade_number']} closed at ${exit_price:.2f}, P&L=${pnl:.2f}")
            send_trade_email(open_trade)
            return True, f"Trade manually closed at ${exit_price:.2f}."

        except Exception as e:
            self._log(f"❌ MANUAL CLOSE ERROR: {str(e)}")
            traceback.print_exc()
            return False, f"An error occurred: {str(e)}"

    def _get_settings_log(self):
        defaults = deepcopy(TradingConfig.DEFAULT_PARAMS)
        for k, v in self.params.items():
            defaults[k] = v

        p = defaults
        return (
            f"Settings: Symbol={p.get('SYMBOL')} | Interval={p.get('INTERVAL')} | "
            f"Capital={p.get('START_CAPITAL')} | SL={p.get('STOP_LOSS_PCT')} | TP={p.get('TAKE_PROFIT_PCT')} | "
            f"TRL={p.get('TRAIL_STOP_PCT')} | BuyProbThresh={p.get('BUY_PROB_THRESHOLD')} | "
            f"ModelType={getattr(self.model, 'model_type', 'N/A')} | ConfidenceThresh={p.get('CONFIDENCE_THRESHOLD')} | "
            f"SignalExitThresh={p.get('SIGNAL_EXIT_THRESHOLD')} | SignalExitConsec={p.get('SIGNAL_EXIT_CONSECUTIVE')} | MinHoldCandles={p.get('MIN_HOLD_CANDLES')}"
        )

    def _log(self, message: str):
        """Adds log message to both in-memory list and file log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} | {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]

        logger.info(message)

    def get_status(self):
        open_trade = next((t for t in self.results['trades'] if t.get('status') == 'OPEN'), None)
        closed_trades = [t for t in self.results['trades'] if t.get('status') == 'CLOSED']        latest_price = self.data_history['Close'].iloc[-1] if not self.data_history.empty and 'Close' in self.data_history.columns else 0
        total_trades = len(self.results['trades'])
        pnl = 0.0
        capital = self.params.get('START_CAPITAL', 10000)
        position = 'Cash'

        if open_trade:
            unrealized_pnl = (latest_price - open_trade['entry_price']) * open_trade['position_size']
            open_trade['unrealized_pnl'] = unrealized_pnl

            pnl = unrealized_pnl
            position = 'Open'
            capital = open_trade.get('capital_after_entry', capital) + pnl
        elif closed_trades:
            pnl = closed_trades[-1]['pnl']
            position = 'Cash'
            capital = self.get_final_capital()

        return {
            'running': self.running,
            'status': 'Running - Live Trading' if self.running else 'Stopped',
            'current_price': latest_price,
            'position': position,
            'pnl': pnl,
            'total_trades': total_trades,
            'capital': capital,
            'trade_details': open_trade if open_trade else (closed_trades[-1] if closed_trades else None)
        }

    def get_final_capital(self):
        final_capital = self.params.get('START_CAPITAL', 10000)
        closed_pnl = sum(t.get('pnl', 0) for t in self.results['trades'] if t.get('status') == 'CLOSED')
        final_capital += closed_pnl

        open_trade = next((t for t in self.results['trades'] if t.get('status') == 'OPEN'), None)
        if open_trade:
            final_capital += open_trade.get('unrealized_pnl', 0)

        return final_capital

    def get_live_data_for_chart(self):
        try:
            if self.data_history.empty:
                return {'priceData': [], 'trades': [], 'openTrade': None}

            price_data = [
                {'x': str(ts), 'y': price}
                for ts, price in self.data_history['Close'].items()
            ]

            closed_trades = [
                {
                    'status': 'CLOSED',
                    'entry_price': t['entry_price'],
                    'exit_price': t['exit_price'],
                    'entry_date': str(pd.to_datetime(t['entry_timestamp'])),
                    'exit_date': str(pd.to_datetime(t['closed_timestamp'])),
                    'pnl': t['pnl']
                }
                for t in self.results['trades'] if t.get('status') == 'CLOSED'
            ]

            open_trade = next((t for t in self.results['trades'] if t.get('status') == 'OPEN'), None)
            open_trade_data = None
            if open_trade:
                open_trade_data = {
                    'entry_price': open_trade['entry_price'],
                    'entry_date': str(pd.to_datetime(open_trade['entry_timestamp'])),
                    'stop_loss': open_trade['stop_loss'],
                    'take_profit': open_trade['take_profit'],
                    'trail_stop_pct': open_trade.get('trail_stop_pct', self.params.get('TRAIL_STOP_PCT')), # Ensure it exists
                    'highest_price': open_trade['highest_price']
                }

            return {
                'priceData': price_data,
                'closedTrades': closed_trades,
                'openTrade': open_trade_data
            }
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            return {'priceData': [], 'closedTrades': [], 'openTrade': None}

# --- Flask App Routes ---
app = Flask(__name__)
backtest_engine = BacktestEngine()
config = TradingConfig.DEFAULT_PARAMS.copy()
settings_manager = SettingsManager()

live_trader = None
live_trading_running = False
latest_backtest_settings = None
current_trained_model = None
latest_model_info = {
    'training_accuracy': 0.0, 'test_accuracy': 0.0, 'validation_accuracy': 0.0, 'hit_rate': 0.0,
    'model_type': 'Not Trained', 'features_count': 0, 'class_distribution': {}
}

# --- MODIFIED: Dynamic filename and folder display ---
@app.route('/')
def dashboard():
    script_path = os.path.abspath(sys.argv[0])
    script_filename = os.path.basename(script_path)
    script_folder = os.path.basename(os.path.dirname(script_path))

    return render_template('dashboard.html',
                           script_filename=script_filename,
                           script_folder=script_folder)

# --- Original routes ---
@app.route('/backtest')
def backtest_page():
    return render_template('backtest.html')

@app.route('/status')
def get_status():
    global latest_model_info
    return jsonify({
        'running': False, 'capital': config.get('START_CAPITAL', 10000), 'current_price': 0, 'position': 0,
        'pnl': 0, 'total_trades': 0, 'model_confidence': 0,
        'model_type': latest_model_info.get('model_type', 'Not Trained'),
        'last_trade': 'None',
        'training_accuracy': latest_model_info.get('training_accuracy', 0.0),
        'test_accuracy': latest_model_info.get('test_accuracy', 0.0),
        'validation_accuracy': latest_model_info.get('validation_accuracy', 0.0),
        'hit_rate': latest_model_info.get('hit_rate', 0.0),
        'features_count': latest_model_info.get('features_count', 0),
        'signal_exit_threshold': config.get('SIGNAL_EXIT_THRESHOLD', 0.25),
        'signal_exit_consecutive': config.get('SIGNAL_EXIT_CONSECUTIVE', 3),
        'min_hold_candles': config.get('MIN_HOLD_CANDLES', 6),
        'total_return_pct': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0
    })

@app.route('/params')
def get_params():
    return jsonify(config)

@app.route('/update_params', methods=['POST'])
def update_params():
    global config
    try:
        data = request.json
        for key, value in data.items():
            if key in config:
                config[key] = value
        return jsonify({'status': 'Parameters updated'})
    except Exception as e:
        return jsonify({'status': f'Error: {str(e)}'})

@app.route('/trades')
def get_trades():
    return jsonify([])

@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    global latest_backtest_settings
    try:
        data = request.json
        params = data.get('params', config)
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')

        latest_backtest_settings = params.copy()

        if backtest_engine.running:
            return jsonify({'status': 'Backtest already running'})

        def run_backtest():
            backtest_engine.run(params, start_date, end_date)

        threading.Thread(target=run_backtest, daemon=True).start()
        return jsonify({'status': 'Backtest started'})

    except Exception as e:
        logger.error(f"Start backtest error: {e}")
        return jsonify({'status': f'Error: {str(e)}'})

@app.route('/stop_backtest', methods=['POST'])
def stop_backtest():
    try:
        backtest_engine.stop()
        return jsonify({'status': 'Backtest stopped'})
    except Exception as e:
        return jsonify({'status': f'Error: {str(e)}'})

@app.route('/backtest_results')
def get_backtest_results():
    try:
        return jsonify(backtest_engine.results)
    except Exception as e:
        logger.error(f"Get results error: {e}")
        return jsonify({'error': str(e)})

@app.route('/backtest_logs')
def get_backtest_logs():
    try:
        return jsonify({'logs': backtest_engine.logs})
    except Exception as e:
        logger.error(f"Get logs error: {e}")
        return jsonify({'logs': [f"Error getting logs: {str(e)}"]})

@app.route('/reset_backtest', methods=['POST'])
def reset_backtest():
    try:
        backtest_engine.stop()
        backtest_engine.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}
        backtest_engine.logs = ['Ready to start backtest...']
        backtest_engine.running = False
        logger.info("Backtest reset successfully")
        return jsonify({'status': 'Backtest reset successfully'})
    except Exception as e:
        logger.error(f"Reset backtest error: {e}")
        return jsonify({'status': f'Error: {str(e)}'})

@app.route('/save_model', methods=['POST'])
def save_current_model():
    global current_trained_model
    try:
        if current_trained_model is None or not current_trained_model.is_trained:
            return jsonify({'status': 'error', 'message': 'No trained model available to save'})
        data = request.json or {}
        custom_name = data.get('filename', None)
        if custom_name:
            if not custom_name.endswith('.pkl'):
                custom_name += '.pkl'
            filepath = f"models/{custom_name}"
        else:
            filepath = None
        saved_path = current_trained_model.save_model(filepath)
        return jsonify({
            'status': 'success', 'message': f'Model saved successfully to {saved_path}', 'filepath': saved_path
        })
    except Exception as e:
        logger.error(f"Save model error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/load_model', methods=['POST'])
def load_saved_model():
    global current_trained_model, latest_model_info
    try:
        data = request.json
        filepath = data.get('filepath')
        if not filepath:
            return jsonify({'status': 'error', 'message': 'No filepath provided'})
        model = MLModel()
        success = model.load_model(filepath)
        if success:
            current_trained_model = model
            model_info = model.get_model_info()
            latest_model_info.update({
                'training_accuracy': model_info.get('training_accuracy', 0.0), 'test_accuracy': model_info.get('test_accuracy', 0.0),
                'validation_accuracy': model_info.get('validation_accuracy', 0.0), 'hit_rate': model_info.get('hit_rate', 0.0),
                'model_type': model_info.get('model_type', 'Unknown'), 'features_count': model_info.get('n_features', 0),
                'class_distribution': model_info.get('class_distribution', {})
            })
            return jsonify({
                'status': 'success', 'message': f'Model loaded successfully from {filepath}', 'model_info': model_info
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to load model'})
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/list_models')
def list_saved_models():
    try:
        models = MLModel.list_saved_models()
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"List models error: {e}")
        return jsonify({'error': str(e)})

@app.route('/delete_model', methods=['POST'])def delete_saved_model():
    try:
        data = request.json
        filepath = data.get('filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'status': 'error', 'message': 'Model file not found'})
        os.remove(filepath)
        return jsonify({'status': 'success', 'message': f'Model deleted: {filepath}'})
    except Exception as e:
        logger.error(f"Delete model error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_settings', methods=['POST'])
def save_current_settings():
    global config, settings_manager
    try:
        data = request.json or {}
        custom_name = data.get('filename', None)
        saved_path = settings_manager.save_settings(config, custom_name)
        return jsonify({
            'status': 'success', 'message': f'Settings saved successfully to {saved_path}', 'filepath': saved_path
        })
    except Exception as e:
        logger.error(f"Save settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/load_settings', methods=['POST'])
def load_saved_settings():
    global config, settings_manager
    try:
        data = request.json
        filepath = data.get('filepath')
        if not filepath:
            return jsonify({'status': 'error', 'message': 'No filepath provided'})
        loaded_settings = settings_manager.load_settings(filepath)
        config.update(loaded_settings)
        return jsonify({
            'status': 'success', 'message': f'Settings loaded successfully from {filepath}', 'settings': loaded_settings
        })
    except Exception as e:
        logger.error(f"Load settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/list_settings')
def list_saved_settings():
    try:
        settings = settings_manager.list_saved_settings()
        return jsonify({'settings': settings})
    except Exception as e:
        logger.error(f"List settings error: {e}")
        return jsonify({'error': str(e)})

@app.route('/delete_settings', methods=['POST'])
def delete_saved_settings():
    try:
        data = request.json
        filepath = data.get('filepath')
        if not filepath:
            return jsonify({'status': 'error', 'message': 'No filepath provided'})
        success = settings_manager.delete_settings(filepath)
        if success:
            return jsonify({'status': 'success', 'message': f'Settings deleted: {filepath}'})
        else:
            return jsonify({'status': 'error', 'message': 'Settings file not found'})
    except Exception as e:
        logger.error(f"Delete settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_backtest_settings', methods=['POST'])
def save_backtest_settings():
    global settings_manager
    try:
        data = request.json or {}
        custom_name = data.get('filename', None)
        params = data.get('params', {})
        if not params:
            return jsonify({'status': 'error', 'message': 'No parameters provided'})
        saved_path = settings_manager.save_settings(params, custom_name)
        return jsonify({
            'status': 'success', 'message': f'Backtest settings saved successfully to {saved_path}', 'filepath': saved_path
        })
    except Exception as e:
        logger.error(f"Save backtest settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/create_project_zip', methods=['POST'])
def create_project_zip():
    try:
        import zipfile
        from io import BytesIO
        from flask import send_file
        data = request.json or {}
        custom_filename = data.get('filename', None)
        if custom_filename:
            if not custom_filename.endswith('.zip'):
                custom_filename += '.zip'
            zip_filename = custom_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"bitcoin_trading_bot_project_{timestamp}.zip"
        zip_buffer = BytesIO()
        files_added = 0
        exclude_dirs = {
            '__pycache__', '.git', '.replit', 'node_modules', '.env', 'venv', '.venv', '.DS_Store', 'Thumbs.db', '.pytest_cache',
            'attached_assets', 'bitcoin_bot_extracted'
        }
        exclude_files = {
            'uv.lock', 'generated-icon.png', 'bitcoin_trading_bot_duplicate.zip', '.replit'
        }
        logger.info(f"🔧 Creating ZIP file: {zip_filename}")
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
            main_files = ['main.py', 'pyproject.toml', 'main5.py'] # Added main5.py for completeness
            for main_file in main_files:
                if os.path.exists(main_file):                    zip_file.write(main_file, main_file)
                    files_added += 1
                    logger.info(f"✅ Added: {main_file}")
            if os.path.exists('templates'):
                for root, dirs, files in os.walk('templates'):
                    for file in files:
                        if file.endswith(('.html', '.css', '.js')):
                            file_path = os.path.join(root, file)
                            archive_path = file_path.replace('\\', '/')
                            zip_file.write(file_path, archive_path)
                            files_added += 1
                            logger.info(f"✅ Added: {archive_path}")
            if os.path.exists('models'):
                for root, dirs, files in os.walk('models'):
                    for file in files:
                        if file.endswith('.pkl'):
                            file_path = os.path.join(root, file)
                            archive_path = file_path.replace('\\', '/')
                            zip_file.write(file_path, archive_path)
                            files_added += 1
                            logger.info(f"✅ Added: {archive_path}")
            if os.path.exists('settings'):
                for root, dirs, files in os.walk('settings'):
                    for file in files:
                        if file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            archive_path = file_path.replace('\\', '/')
                            zip_file.write(file_path, archive_path)
                            files_added += 1
                            logger.info(f"✅ Added: {archive_path}")
            for file in os.listdir('.'):
                if (file.endswith('.py') and file not in exclude_files and file != 'main.py' and file != 'main5.py' and os.path.isfile(file)):
                    zip_file.write(file, file)
                    files_added += 1
                    logger.info(f"✅ Added: {file}")
        zip_buffer.seek(0)
        if files_added == 0:
            logger.error("❌ No files were added to ZIP")
            return jsonify({'status': 'error', 'message': 'No files found to zip'}), 400
        logger.info(f"✅ ZIP created successfully: {zip_filename} ({files_added} files)")
        return send_file(
            zip_buffer, as_attachment=True, download_name=zip_filename, mimetype='application/zip'
        )
    except Exception as e:
        logger.error(f"❌ ZIP creation error: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_latest_settings')
def get_latest_settings():
    global latest_backtest_settings
    try:
        if latest_backtest_settings:
            return jsonify({ 'status': 'success', 'settings': latest_backtest_settings })
        else:
            return jsonify({ 'status': 'error', 'message': 'No backtest settings found. Please run a backtest first.' })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/start_live_trading', methods=['POST'])
def start_live_trading():
    global live_trader, live_trading_running, current_trained_model
    try:
        data = request.json
        params = data.get('params', {})
        if live_trading_running:
            return jsonify({'status': 'error', 'message': 'Live trading already running'})        if not current_trained_model or not current_trained_model.is_trained:
            return jsonify({'status': 'error', 'message': 'No trained model available. Please run a backtest first.'})

        live_trader = LiveTrader()
        live_trader.start(params, current_trained_model)
        live_trading_running = True
        logger.info("Live trading started")
        return jsonify({ 'status': 'success', 'message': 'Live trading started successfully' })
    except Exception as e:
        logger.error(f"Start live trading error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_live_trading', methods=['POST'])
def stop_live_trading():
    global live_trader, live_trading_running
    try:
        if live_trader:
            live_trader.stop()
        live_trading_running = False
        logger.info("Live trading stopped")
        return jsonify({ 'status': 'success', 'message': 'Live trading stopped' })
    except Exception as e:
        logger.error(f"Stop live trading error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/manual_close_trade', methods=['POST'])
def manual_close_trade():
    global live_trader, live_trading_running
    try:
        if live_trader and live_trading_running:
            success, message = live_trader.manual_close_trade()
            if success:
                return jsonify({'status': 'success', 'message': message})
            else:
                return jsonify({'status': 'error', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': 'Live trading is not active.'})
    except Exception as e:
        logger.error(f"Manual close error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset_live_trading', methods=['POST'])
def reset_live_trading():
    global live_trader, live_trading_running
    try:
        if live_trader and live_trading_running:
            live_trader.stop()

        live_trader = None
        live_trading_running = False

        logger.info("Live trading session reset successfully")

        return jsonify({'status': 'success', 'message': 'Live trading session reset successfully.'})
    except Exception as e:
        logger.error(f"Reset live trading error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/live_trading_status')
def get_live_trading_status():
    global live_trader, live_trading_running
    try:
        if live_trader and live_trading_running:
            status = live_trader.get_status()            status['trades'] = live_trader.results['trades']
            return jsonify(status)
        else:
            return jsonify({
                'running': False, 'status': 'Stopped', 'current_price': 0, 'position': 'Cash', 'pnl': 0,
                'total_trades': 0, 'capital': 0, 'trades': []
            })
    except Exception as e:
        logger.error(f"Get live trading status error: {e}")
        return jsonify({'running': False, 'error': str(e)})

@app.route('/live_trading_logs')
def get_live_trading_logs():
    global live_trader
    try:
        if live_trader and hasattr(live_trader, 'logs'):
            return jsonify({'logs': live_trader.logs})
        else:
            return jsonify({'logs': ['No live trading session active']})
    except Exception as e:
        logger.error(f"Get live trading logs error: {e}")
        return jsonify({'logs': [f"Error getting logs: {str(e)}"]})

@app.route('/live_trading_chart_data')
def get_live_trading_chart_data():
    global live_trader
    if live_trader and live_trader.running:
        data = live_trader.get_live_data_for_chart()
        return jsonify(data)
    else:
        return jsonify({'priceData': [], 'closedTrades': [], 'openTrade': None})

@app.route('/ping')
def ping():
    try:
        return jsonify({
            'status': 'alive', 'timestamp': datetime.now().isoformat(),
            'server': 'Bitcoin Trading Bot', 'uptime': 'running'
        })
    except Exception as e:
        return jsonify({'status': 'alive', 'error': str(e)})

if __name__ == '__main__':
    logger.info("🚀 Starting Bitcoin Trading Bot Server")
    app.run(host='0.0.0.0', port=5000, de

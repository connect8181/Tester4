
     1
     2  import importlib
     3  import subprocess
     4  import sys
     5  import os
     6  import json
     7  import time
     8  import threading
     9  import logging
    10  import pandas as pd
    11  import numpy as np
    12  import traceback
    13  from datetime import datetime, timedelta
    14  from typing import Dict, List, Optional, Tuple
    15  import joblib
    16  from copy import deepcopy
    17
    18  import smtplib
    19  from email.mime.multipart import MIMEMultipart
    20  from email.mime.text import MIMEText
    21
    22  EMAIL_ADDRESS = "connect18181g@gmail.com"
    23  EMAIL_PASSWORD = "xdvo ethw pepz aqxl"
    24  EMAIL_RECEIVER = "connect18181g@gmail.com"
    25
    26  def send_trade_email(trade_info):
    27      try:
    28          msg = MIMEMultipart()
    29          msg['From'] = EMAIL_ADDRESS
    30          msg['To'] = EMAIL_RECEIVER
    31          msg['Subject'] = f"Live Trade Notification: {trade_info.get('exit_reason', trade_info.get('status',''))}"
    32          body = f"""
    33  Trade Number: {trade_info.get('trade_number', '')}
    34  Type: {trade_info.get('status', '')}
    35  Symbol: {trade_info.get('symbol', '')} | Interval: {trade_info.get('interval', '')}
    36  Entry Price: {trade_info.get('entry_price', 0):.2f}
    37  Exit Price: {trade_info.get('exit_price', 0):.2f}
    38  Position Size: {trade_info.get('position_size', 0):.6f}
    39  P&L: {trade_info.get('pnl', 0):.2f}
    40  Capital After Trade: {trade_info.get('capital_after_entry', 0):.2f}
    41  Timestamp: {trade_info.get('closed_timestamp', trade_info.get('entry_timestamp', ''))}
    42  Model Type: {trade_info.get('model_type', '')}
    43  Stop Loss: {trade_info.get('stop_loss', 0):.2f}
    44  Take Profit: {trade_info.get('take_profit', 0):.2f}
    45  Highest Price Since Entry: {trade_info.get('highest_price', 0):.2f}
    46          """
    47          msg.attach(MIMEText(body, 'plain'))
    48          server = smtplib.SMTP('smtp.gmail.com', 587)
    49          server.starttls()
    50          server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    51          server.send_message(msg)
    52          server.quit()
    53      except Exception as e:
    54          print(f"❌ Fehler beim Senden der E-Mail: {e}")
    55
    56
    57  # --- Installation and Imports ---
    58  def install_if_missing(package_name, pip_name=None):
    59      try:
    60          importlib.import_module(package_name)
    61      except ImportError:
    62          subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package_name])
    63
    64  packages = [
    65      ('yfinance', None),
    66      ('pandas', None),
    67      ('numpy', None),
    68      ('sklearn', 'scikit-learn'),
    69      ('flask', None),
    70      ('binance', 'python-binance'),
    71      ('imbalanced_learn', 'imbalanced-learn'),
    72      ('requests', None)
    73  ]
    74
    75  for pkg, pip_name in packages:
    76      install_if_missing(pkg, pip_name)    77
    78  import yfinance as yf
    79  from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    80  from sklearn.preprocessing import StandardScaler
    81  from sklearn.model_selection import train_test_split
    82  from sklearn.metrics import classification_report, accuracy_score
    83  try:
    84      from imblearn.over_sampling import RandomOverSampler
    85  except ImportError:
    86      RandomOverSampler = None
    87  from flask import Flask, render_template, jsonify, request, send_file
    88  from binance.client import Client
    89  import requests
    90  import json
    91  import joblib
    92  import zipfile
    93  from io import BytesIO
    94
    95  logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    96  logger = logging.getLogger(__name__)
    97
    98  # --- Configuration and Settings ---
    99  class TradingConfig:
   100      """Trading configuration and parameters"""
   101      DEFAULT_PARAMS = {
   102          'SYMBOL': 'BTCUSDT',
   103          'INTERVAL': '5m',
   104          'TRAINING_PERIOD': '60d',
   105          'BACKTEST_PERIOD': '30d',
   106          'START_CAPITAL': 10000,
   107          'MAKER_FEE_RATE': 0.001,
   108          'TAKER_FEE_RATE': 0.001,
   109          'STOP_LOSS_PCT': 0.05,
   110          'TAKE_PROFIT_PCT': 0.008,
   111          'TRAIL_STOP_PCT': 0.02,
   112          'BUY_PROB_THRESHOLD': 0.9,
   113          'MODEL_TYPE': 'HistGradientBoosting',
   114          'CONFIDENCE_THRESHOLD': 0.7,
   115          'FUTURE_WINDOW': 24,
   116          'USE_OVERSAMPLING': True,
   117          'TEST_SIZE': 0.3,
   118          'SIGNAL_EXIT_THRESHOLD': 0.25,
   119          'SIGNAL_EXIT_CONSECUTIVE': 3,   120          'MIN_HOLD_CANDLES': 6
   121      }
   122
   123  class SettingsManager:
   124      """Manage saving and loading of trading parameters"""
   125
   126      def __init__(self, settings_dir: str = "settings"):
   127          self.settings_dir = settings_dir
   128          os.makedirs(settings_dir, exist_ok=True)
   129
   130      def save_settings(self, params: Dict, name: str = None) -> str:
   131          """Save parameters to a JSON file"""
   132          try:
   133              if name is None:
   134                  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   135                  name = f"settings_{timestamp}"
   136
   137              if not name.endswith('.json'):
   138                  name += '.json'
   139
   140              filepath = os.path.join(self.settings_dir, name)
   141
   142              settings_data = deepcopy(params)
   143              settings_data['_metadata'] = {
   144                  'saved_at': datetime.now().isoformat(),
   145                  'name': name,
   146                  'version': '1.0'
   147              }
   148
   149              with open(filepath, 'w') as f:
   150                  json.dump(settings_data, f, indent=2)
   151
   152              logger.info(f"Settings saved to: {filepath}")
   153              return filepath
   154
   155          except Exception as e:
   156              logger.error(f"Error saving settings: {e}")
   157              raise e
   158
   159      def load_settings(self, filepath: str) -> Dict:
   160          """Load parameters from a JSON file"""
   161          try:
   162              if not os.path.exists(filepath):
   163                  raise FileNotFoundError(f"Settings file not found: {filepath}")
   164
   165              with open(filepath, 'r') as f:
   166                  settings_data = json.load(f)
   167
   168              if '_metadata' in settings_data:
   169                  del settings_data['_metadata']
   170
   171              logger.info(f"Settings loaded from: {filepath}")
   172              return settings_data
   173
   174          except Exception as e:
   175              logger.error(f"Error loading settings: {e}")
   176              raise e
   177
   178      def list_saved_settings(self) -> List[Dict]:
   179          """List all saved settings with metadata"""
   180          try:
   181              settings = []
   182              if not os.path.exists(self.settings_dir):
   183                  return settings
   184
   185              for filename in os.listdir(self.settings_dir):
   186                  if filename.endswith('.json'):
   187                      filepath = os.path.join(self.settings_dir, filename)
   188                      try:
   189                          with open(filepath, 'r') as f:
   190                              data = json.load(f)
   191
   192                          metadata = data.get('_metadata', {})
   193                          settings.append({
   194                              'filename': filename,
   195                              'filepath': filepath,
   196                              'name': metadata.get('name', filename),
   197                              'saved_at': metadata.get('saved_at', 'Unknown'),
   198                              'symbol': data.get('SYMBOL', 'Unknown'),
   199                              'model_type': data.get('MODEL_TYPE', 'Unknown'),
   200                              'start_capital': data.get('START_CAPITAL', 0)
   201                          })
   202                      except Exception as e:
   203                          logger.error(f"Error reading settings file {filename}: {e}")
   204                          continue
   205
   206              settings.sort(key=lambda x: x['saved_at'], reverse=True)
   207              return settings
   208
   209          except Exception as e:
   210              logger.error(f"Error listing settings: {e}")
   211              return []
   212
   213      def delete_settings(self, filepath: str) -> bool:
   214          """Delete a settings file"""
   215          try:
   216              if os.path.exists(filepath):
   217                  os.remove(filepath)
   218                  logger.info(f"Settings deleted: {filepath}")
   219                  return True
   220              return False
   221          except Exception as e:
   222              logger.error(f"Error deleting settings: {e}")
   223              return False
   224
   225  # --- Data Provision ---
   226  class DataProvider:
   227      """Handles data fetching from various sources"""
   228
   229      @staticmethod
   230      def get_binance_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
   231          """Fetch historical data from Binance API"""
   232          try:
   233              client = Client()
   234              interval_map = {
   235                  '1m': Client.KLINE_INTERVAL_1MINUTE,
   236                  '5m': Client.KLINE_INTERVAL_5MINUTE,
   237                  '15m': Client.KLINE_INTERVAL_15MINUTE,
   238                  '1h': Client.KLINE_INTERVAL_1HOUR,
   239                  '1d': Client.KLINE_INTERVAL_1DAY
   240              }
   241
   242              binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_5MINUTE)
   243              klines = client.get_historical_klines(symbol, binance_interval, start_date, end_date)
   244
   245              if not klines:
   246                  return pd.DataFrame()   247
   248              df = pd.DataFrame(klines, columns=[
   249                  'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
   250                  'close_time', 'quote_asset_volume', 'number_of_trades',
   251                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
   252              ])
   253
   254              df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
   255              df.set_index('timestamp', inplace=True)
   256              for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
   257                  df[col] = df[col].astype(float)
   258
   259              return df[['Open', 'High', 'Low', 'Close', 'Volume']]
   260
   261          except Exception as e:
   262              logger.error(f"Binance API error: {e}")
   263              return pd.DataFrame()
   264
   265      @staticmethod
   266      def get_yfinance_data(symbol: str, period: str = '60d', interval: str = '5m') -> pd.DataFrame:
   267          """Fetch data from Yahoo Finance"""
   268          try:
   269              yf_symbol = 'BTC-USD' if symbol == 'BTCUSDT' else symbol
   270              df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
   271              if df.empty:
   272                  return pd.DataFrame()   273
   274              if isinstance(df.columns, pd.MultiIndex):
   275                  df.columns = [col[0] for col in df.columns]
   276
   277              required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
   278              available_cols = [col for col in required_cols if col in df.columns]
   279              if len(available_cols) == len(required_cols):
   280                  df = df[required_cols]
   281              else:
   282                  logger.error(f"Missing columns in yfinance data. Available: {df.columns.tolist()}")
   283                  return pd.DataFrame()   284
   285              return df
   286          except Exception as e:
   287              logger.error(f"YFinance error: {e}")
   288              return pd.DataFrame()
   289
   290      @staticmethod
   291      def get_training_data(symbol: str, period: str = '60d', interval: str = '5m') -> pd.DataFrame:
   292          """Get data specifically for model training"""
   293          logger.info(f"🔧 TRAINING DATA: Using Yahoo Finance")
   294          return DataProvider.get_yfinance_data(symbol, period, interval)
   295
   296      @staticmethod
   297      def get_backtest_data_with_period(symbol: str, period: str = '30d', interval: str = '5m') -> pd.DataFrame:
   298          """Get data specifically for backtesting with configurable period"""
   299          data = DataProvider.get_binance_data(symbol, interval,
   300              (datetime.now() - timedelta(days=int(period.rstrip('d')))).strftime('%Y-%m-%d'),
   301              datetime.now().strftime('%Y-%m-%d'))
   302          if data.empty:
   303              yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
   304              data = DataProvider.get_yfinance_data(yf_symbol, period, interval)
   305          return data
   306
   307      @staticmethod
   308      def get_live_data(symbol: str, interval: str = '5m') -> pd.DataFrame:
   309          """
   310          Get live data for live trading - uses same source as training.
   311          Fetches the last 5 days (hardcoded in yfinance) to ensure enough
   312          history for feature extraction (like RSI 14).
   313          """
   314          logger.info(f"📡 LIVE TRADING: Getting latest data (5d history)")
   315          try:
   316              yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
   317              # Hardcoded '5d' period to ensure enough lookback data for TA features   318              df = DataProvider.get_yfinance_data(yf_symbol, '5d', interval)
   319              if not df.empty:
   320                  logger.info(f"✅ Live data: {len(df)} candles from Yahoo Finance")   321                  return df
   322              else:
   323                  raise Exception("Empty dataframe from Yahoo Finance")
   324          except Exception as yf_error:   325              logger.error(f"❌ Yahoo Finance error: {yf_error}")
   326          return pd.DataFrame()
   327
   328  # --- Technical Analysis ---
   329  class TechnicalAnalysis:
   330      """Technical analysis and feature extraction"""
   331      @staticmethod
   332      def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
   333          """Calculate RSI indicator"""   334          try:
   335              delta = series.diff()
   336              gain = delta.clip(lower=0)
   337              loss = -delta.clip(upper=0)
   338              avg_gain = gain.rolling(window=window, min_periods=1).mean()
   339              avg_loss = loss.rolling(window=window, min_periods=1).mean()
   340              rs = avg_gain / (avg_loss + 1e-10)
   341              rsi = 100 - (100 / (1 + rs))
   342              return rsi.fillna(50)
   343          except Exception as e:
   344              logger.error(f"RSI calculation error: {e}")
   345              return pd.Series([50] * len(series), index=series.index)
   346
   347      @staticmethod
   348      def extract_features(df: pd.DataFrame, future_window: int = 24, is_live_trading: bool = False) -> pd.DataFrame:
   349          """Extract technical features using simplified approach"""
   350          try:
   351              # Increased minimum length to ensure RSI (window=14) can be calculated   352              MIN_DATA_LENGTH = 20
   353
   354              if df.empty or len(df) < MIN_DATA_LENGTH:
   355                  if is_live_trading:
   356                      logger.warning(f"Insufficient data for feature extraction (Min {MIN_DATA_LENGTH}, Got {len(df)})")
   357                  return pd.DataFrame()   358
   359              df = df.copy()
   360
   361              if is_live_trading:
   362                  logger.info(f"🔧 LIVE FEATURE EXTRACTION:")
   363                  logger.info(f"   Input data shape: {df.shape}")
   364                  logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
   365                  logger.info(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
   366
   367              df['returns'] = df['Close'].pct_change()
   368              df['volatility'] = df['returns'].rolling(window=5).std()
   369              df['rsi'] = TechnicalAnalysis.compute_rsi(df['Close'])
   370
   371              # This is the critical line that removes rows with NaN features (e.g., the first 14 rows due to RSI)
   372              df = df.dropna()
   373
   374              if df.empty:
   375                  logger.error("All data was NaN after feature extraction")
   376                  return pd.DataFrame()   377
   378              # Ensure at least 1 candle with features remains for live prediction
   379              # --- FIX: SYNTAX ERROR WAS HERE ---
   380              if is_live_trading and len(df) < 1:
   381                  logger.error("❌ Not enough data points remain after feature extraction.")
   382                  return pd.DataFrame()   383              # -----------------------------------
   384
   385
   386              if is_live_trading:
   387                  logger.info(f"   ✅ Final features shape: {df.shape}")
   388                  latest = df.iloc[-1]
   389                  logger.info(f"   ✅ Latest: returns={latest['returns']:.6f}, volatility={latest['volatility']:.6f}, rsi={latest['rsi']:.2f}")
   390
   391              return df
   392
   393          except Exception as e:
   394              logger.error(f"Feature extraction error: {e}")
   395              traceback.print_exc()
   396              return pd.DataFrame()
   397
   398  # --- ML Model ---
   399  class MLModel:
   400      """Enhanced machine learning model for trading predictions"""
   401
   402      def __init__(self, model_type: str = 'HistGradientBoosting'):
   403          self.model_type = model_type
   404          self.model = self._get_model(model_type)
   405          self.scaler = StandardScaler()
   406          self.features = ['returns', 'volatility', 'rsi']
   407          self.is_trained = False
   408          self.training_accuracy = 0.0
   409          self.test_accuracy = 0.0
   410          self.validation_accuracy = 0.0
   411          self.hit_rate = 0.0
   412          self.class_distribution = {}
   413
   414      def _get_model(self, model_type: str):
   415          """Get the appropriate model based on type"""
   416          if model_type == 'HistGradientBoosting':
   417              return HistGradientBoostingClassifier(
   418                  max_iter=200,
   419                  learning_rate=0.1,
   420                  max_depth=6,
   421                  random_state=42
   422              )
   423          elif model_type == 'RandomForest':
   424              return RandomForestClassifier(
   425                  n_estimators=100,
   426                  max_depth=15,
   427                  min_samples_split=5,
   428                  min_samples_leaf=2,
   429                  random_state=42
   430              )
   431          else:
   432              logger.warning(f"Unknown model type {model_type}, using HistGradientBoosting")
   433              return HistGradientBoostingClassifier(max_iter=200, random_state=42)
   434
   435      def train(self, df: pd.DataFrame, future_window: int = 24, take_profit_pct: float = 0.008,
   436                use_oversampling: bool = True, test_size: float = 0.3, training_period: str = '60d'):
   437          """
   438          Training method for the model.
   439          """
   440          try:
   441              if df.empty or len(df) < 200:
   442                  logger.error(f"Insufficient data for training: {len(df)} rows")
   443                  return 0.0
   444
   445              df = df.copy()
   446
   447              df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
   448
   449              df.dropna(inplace=True)
   450
   451              target_dist = df['target'].value_counts(normalize=True)
   452              self.class_distribution = target_dist.to_dict()
   453              logger.info(f"Target distribution: {self.class_distribution}")
   454
   455              X = df[self.features]
   456              y = df['target']
   457
   458              if len(y.unique()) < 2:
   459                  logger.error("Target variable has only one class")
   460                  return 0.0
   461
   462              X_scaled = self.scaler.fit_transform(X)
   463
   464              if use_oversampling and len(y.unique()) == 2 and RandomOverSampler is not None:
   465                  try:
   466                      ros = RandomOverSampler(random_state=42)
   467                      X_res, y_res = ros.fit_resample(X_scaled, y)
   468                      logger.info(f"Applied oversampling: {len(X_scaled)} -> {len(X_res)} samples")
   469                  except Exception as e:
   470                      logger.warning(f"Oversampling failed, using original data: {e}")
   471                      X_res, y_res = X_scaled, y
   472              else:
   473                  if RandomOverSampler is None:
   474                      logger.warning("RandomOverSampler not available, using original data")
   475                  X_res, y_res = X_scaled, y
   476
   477              logger.info(f"Training {self.model_type} model...")
   478              self.model.fit(X_res, y_res)
   479              self.is_trained = True
   480
   481              self.training_accuracy = self.model.score(X_scaled, y)
   482
   483              X_train, X_test, y_train, y_test = train_test_split(
   484                  X_scaled, y, test_size=test_size, shuffle=False, random_state=42
   485              )
   486
   487              self.test_accuracy = self.model.score(X_test, y_test)
   488              self.validation_accuracy = self.test_accuracy
   489              self.hit_rate = self.test_accuracy
   490
   491              logger.info(f"=== Training Results ===")
   492              logger.info(f"📊 Model trained with Accuracy: {self.training_accuracy:.4f}")
   493              logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
   494              logger.info(f"Hit Rate: {self.hit_rate:.4f}")
   495              logger.info(f"Model Type: {self.model_type}")
   496              logger.info(f"Features Used: {len(self.features)} - {self.features}")
   497
   498              return self.training_accuracy
   499          except Exception as e:
   500              logger.error(f"Model training error: {e}")
   501              traceback.print_exc()
   502              return 0.0
   503
   504      def predict(self, features: np.ndarray, is_live_trading: bool = False) -> Tuple[float, float]:
   505          """Make prediction and return probability and confidence"""
   506          try:
   507              if not self.is_trained or self.model is None:
   508                  logger.warning("Model not trained or None")
   509                  return 0.5, 0.5
   510
   511              if len(features) != len(self.features):
   512                  logger.error(f"Feature mismatch: expected {len(self.features)}, got {len(features)}")
   513                  return 0.5, 0.5
   514
   515              if is_live_trading:
   516                  logger.info(f"🔍 LIVE PREDICTION:")
   517                  logger.info(f"   Features: {[f'{val:.6f}' for val in features]}")
   518
   519                  nan_check = [np.isnan(val) or np.isinf(val) for val in features]
   520                  if any(nan_check):
   521                      logger.error(f"     INVALID FEATURES: {nan_check}")
   522                      return 0.5, 0.5
   523
   524              X_scaled = self.scaler.transform([features])
   525
   526              if is_live_trading:
   527                  logger.info(f"   Scaled: {[f'{val:.6f}' for val in X_scaled[0]]}")   528
   529              probabilities = self.model.predict_proba(X_scaled)[0]
   530
   531              if is_live_trading:
   532                  logger.info(f"   Raw probabilities: {[f'{p:.6f}' for p in probabilities]}")
   533
   534              if len(probabilities) > 1:
   535                  probability = probabilities[1]
   536                  confidence = max(probabilities)
   537              else:
   538                  probability = 0.5
   539                  confidence = 0.5
   540
   541              if is_live_trading:
   542                  logger.info(f"   ✅ Buy prob: {probability:.6f}, Confidence: {confidence:.6f}")
   543
   544              return float(probability), float(confidence)
   545
   546          except Exception as e:
   547              logger.error(f"Prediction error: {e}")
   548              return 0.5, 0.5
   549
   550      def get_feature_importance(self) -> Dict[str, float]:
   551          """Get feature importance if available"""
   552          try:
   553              if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
   554                  return {}
   555
   556              importance_dict = dict(zip(self.features, self.model.feature_importances_))
   557              return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
   558          except Exception as e:
   559              logger.error(f"Feature importance error: {e}")
   560              return {}
   561
   562      def save_model(self, filepath: str = None) -> str:
   563          """Save model and scaler to file"""
   564          try:
   565              if not self.is_trained:
   566                  raise ValueError("Model must be trained before saving")
   567
   568              if filepath is None:
   569                  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   570                  filepath = f"models/trading_model_{self.model_type}_{timestamp}.pkl"
   571
   572              os.makedirs(os.path.dirname(filepath), exist_ok=True)
   573
   574              model_data = {
   575                  'model': self.model,
   576                  'scaler': self.scaler,
   577                  'model_type': self.model_type,
   578                  'features': self.features,
   579                  'training_accuracy': self.training_accuracy,
   580                  'test_accuracy': self.test_accuracy,
   581                  'validation_accuracy': self.validation_accuracy,
   582                  'hit_rate': self.hit_rate,
   583                  'class_distribution': self.class_distribution,
   584                  'saved_at': datetime.now().isoformat()
   585              }
   586
   587              joblib.dump(model_data, filepath)
   588              logger.info(f"Model saved successfully to: {filepath}")
   589              return filepath
   590
   591          except Exception as e:
   592              logger.error(f"Error saving model: {e}")
   593              raise e
   594
   595      def load_model(self, filepath: str) -> bool:
   596          """Load model and scaler from file"""
   597          try:
   598              if not os.path.exists(filepath):
   599                  raise FileNotFoundError(f"Model file not found: {filepath}")
   600
   601              model_data = joblib.load(filepath)
   602
   603              self.model = model_data['model']
   604              self.scaler = model_data['scaler']
   605              self.model_type = model_data['model_type']
   606              self.features = model_data['features']
   607              self.training_accuracy = model_data.get('training_accuracy', 0.0)
   608              self.test_accuracy = model_data.get('test_accuracy', 0.0)
   609              self.validation_accuracy = model_data.get('validation_accuracy', 0.0)
   610              self.hit_rate = model_data.get('hit_rate', 0.0)
   611              self.class_distribution = model_data.get('class_distribution', {})
   612              self.is_trained = True
   613
   614              logger.info(f"Model loaded successfully from: {filepath}")
   615              logger.info(f"Model type: {self.model_type}, Accuracy: {self.training_accuracy:.4f}")
   616              return True
   617
   618          except Exception as e:
   619              logger.error(f"Error loading model: {e}")
   620              return False
   621
   622      @staticmethod
   623      def list_saved_models(models_dir: str = "models") -> List[Dict]:
   624          """List all saved models with their metadata"""
   625          try:
   626              if not os.path.exists(models_dir):
   627                  return []
   628
   629              models = []
   630              for filename in os.listdir(models_dir):
   631                  if filename.endswith('.pkl'):
   632                      filepath = os.path.join(models_dir, filename)
   633                      try:
   634                          model_data = joblib.load(filepath)
   635                          models.append({
   636                              'filename': filename,
   637                              'filepath': filepath,
   638                              'model_type': model_data.get('model_type', 'Unknown'),   639                              'training_accuracy': model_data.get('training_accuracy', 0.0),
   640                              'saved_at': model_data.get('saved_at', 'Unknown'),
   641                              'features': model_data.get('features', [])
   642                          })
   643                      except Exception as e:
   644                          logger.error(f"Error reading model file {filename}: {e}")
   645                          continue
   646
   647              models.sort(key=lambda x: x['saved_at'], reverse=True)
   648              return models
   649
   650          except Exception as e:
   651              logger.error(f"Error listing models: {e}")
   652              return []
   653
   654      def get_model_info(self) -> Dict:   655          """Get comprehensive model information"""
   656          return {
   657              'model_type': self.model_type,
   658              'is_trained': self.is_trained,
   659              'training_accuracy': self.training_accuracy,
   660              'test_accuracy': self.test_accuracy,
   661              'validation_accuracy': self.validation_accuracy,
   662              'hit_rate': self.hit_rate,
   663              'features': self.features,
   664              'n_features': len(self.features),
   665              'class_distribution': self.class_distribution,
   666              'feature_importance': self.get_feature_importance()
   667          }
   668
   669  class BacktestEngine:
   670      """Backtesting engine"""
   671
   672      def __init__(self):
   673          self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}
   674          self.logs = []
   675          self.running = False
   676
   677      def run(self, params: Dict, start_date: str, end_date: str):
   678          """Run backtest"""
   679          self.running = True
   680          self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': True}
   681          self.logs = []
   682
   683          try:
   684              self._log(f"🚀 Starting backtest: {params['SYMBOL']} ({start_date} to {end_date})")
   685
   686              backtest_period = params.get('BACKTEST_PERIOD', '30d')
   687              data = self._get_backtest_data(params['SYMBOL'], params['INTERVAL'], start_date, end_date, backtest_period)
   688              if data.empty:
   689                  self._log("❌ No data available")
   690                  return
   691
   692              self._log(f"✅ Retrieved {len(data)} candles using {backtest_period} period")
   693
   694              if len(data) > 5000:
   695                  data = data.tail(5000)
   696                  self._log(f"📊 Limited to {len(data)} candles for performance")
   697
   698              self._log("🔧 Extracting technical features...")
   699              data = TechnicalAnalysis.extract_features(data, params['FUTURE_WINDOW'])
   700
   701              if data.empty or len(data) < 200:
   702                  self._log("❌ Insufficient data after feature extraction")
   703                  return
   704
   705              self._log(f"✅ Features extracted from {len(data)} candles")
   706
   707              model = MLModel(params['MODEL_TYPE'])
   708              training_period = params.get('TRAINING_PERIOD', '60d')
   709              accuracy = model.train(
   710                  data,
   711                  params['FUTURE_WINDOW'],
   712                  params['TAKE_PROFIT_PCT'],
   713                  params.get('USE_OVERSAMPLING', True),
   714                  params.get('TEST_SIZE', 0.3),
   715                  training_period
   716              )
   717
   718              global current_trained_model
   719              current_trained_model = model
   720              global latest_model_info
   721              model_info = model.get_model_info()
   722              latest_model_info = {
   723                  'training_accuracy': model_info.get('training_accuracy', 0.0),
   724                  'test_accuracy': model_info.get('test_accuracy', 0.0),
   725                  'validation_accuracy': model_info.get('validation_accuracy', 0.0),   726                  'hit_rate': model_info.get('hit_rate', 0.0),
   727                  'model_type': model_info.get('model_type', 'Unknown'),
   728                  'features_count': model_info.get('n_features', 0),
   729                  'class_distribution': model_info.get('class_distribution', {})
   730              }
   731              self._log(f"📊 Model Info: {model_info['model_type']}, Features: {model_info['n_features']}")
   732
   733              split_idx = int(len(data) * (1 - params.get('TEST_SIZE', 0.3)))
   734              test_data = data.iloc[split_idx:]
   735              self._log(f"📚 Training: {len(data.iloc[:split_idx])} candles, Testing: {len(test_data)} candles")
   736
   737              if accuracy == 0.0:
   738                  self._log("❌ Model training failed")
   739                  return
   740
   741              self._log(f"🤖 Model trained with {accuracy:.2%} accuracy")
   742
   743              self._simulate_trading(model, test_data, params)
   744
   745          except Exception as e:
   746              self._log(f"❌ Backtest error: {str(e)}")
   747              logger.error(f"Backtest error: {e}")
   748              traceback.print_exc()
   749          finally:
   750              self.running = False
   751              self.results['running'] = False
   752
   753      def _get_backtest_data(self, symbol: str, interval: str, start_date: str, end_date: str, period: str = '30d') -> pd.DataFrame:
   754          """Get data for backtesting"""
   755          has_dates = start_date and end_date and start_date.strip() and end_date.strip()
   756
   757          if has_dates:
   758              try:
   759                  start_dt = datetime.strptime(start_date, '%Y-%m-%d')
   760                  end_dt = datetime.strptime(end_date, '%Y-%m-%d')
   761                  days_diff = (end_dt - start_dt).days
   762                  self._log(f"📅 Using specific date range: {start_date} to {end_date} ({days_diff} days)")
   763
   764                  data = DataProvider.get_binance_data(symbol, interval, start_date, end_date)
   765
   766                  if data.empty:
   767                      calculated_period = f"{max(days_diff, 7)}d"
   768                      self._log(f"🔄 Binance unavailable, using Yahoo Finance with {calculated_period}...")
   769                      yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
   770                      data = DataProvider.get_yfinance_data(yf_symbol, calculated_period, interval)
   771
   772                      if not data.empty:
   773                          try:
   774                              start_dt_pd = pd.to_datetime(start_date)
   775                              end_dt_pd = pd.to_datetime(end_date) + pd.Timedelta(days=1)
   776                              data = data[(data.index >= start_dt_pd) & (data.index < end_dt_pd)]
   777                              self._log(f"🤔 Filtered to exact date range: {len(data)} candles")
   778                          except Exception as e:
   779                              self._log(f"⚠️ Date filtering failed: {e}")
   780                  return data
   781
   782              except Exception as e:
   783                  self._log(f"📅 Date parsing error: {e}, falling back to period: {period}")
   784
   785          self._log(f"📅 Using backtest period: {period} (no specific dates)")
   786          data = DataProvider.get_backtest_data_with_period(symbol, period, interval)
   787
   788          return data
   789
   790      def _simulate_trading(self, model: MLModel, data: pd.DataFrame, params: Dict):   791          """Simulate trading on historical data"""
   792          try:
   793              capital = params['START_CAPITAL']
   794              position = 0
   795              btc_amount = 0
   796              entry_price = 0
   797              trades = []
   798              equity_curve = []
   799              total_fees = 0
   800              open_trade_levels = {}
   801              highest_price_since_entry = 0
   802              self._log(f"💰 Starting simulation with ${capital:,.2f}")
   803
   804              chunk_size = 100
   805              total_processed = 0
   806
   807              trade_entry_timestamp = None
   808              buy_probability = 0.0
   809
   810              # The stop-loss is triggered on the low of the candle, not the close
   811              # to prevent it from being "skipped" during a sharp drop.
   812
   813              for chunk_start in range(0, len(data), chunk_size):
   814                  if not self.running:
   815                      break
   816
   817                  chunk_end = min(chunk_start + chunk_size, len(data))
   818                  chunk_data = data.iloc[chunk_start:chunk_end]
   819
   820                  for i, (timestamp, candle) in enumerate(chunk_data.iterrows()):
   821                      if not self.running:
   822                          break
   823
   824                      try:
   825                          price = float(candle['Close'])
   826                          equity = capital if position == 0 else (btc_amount * price)
   827                          equity_curve.append({'timestamp': str(timestamp), 'equity': equity})
   828
   829                          try:
   830                              available_features = [f for f in model.features if f in candle.index]
   831                              if len(available_features) != len(model.features):
   832                                  logger.warning(f"Missing features: {set(model.features) - set(available_features)}")
   833                                  continue
   834                              features = candle[model.features].values
   835                              prob, confidence = model.predict(features)
   836                          except Exception as e:
   837                              logger.error(f"Feature extraction error for prediction: {e}")
   838                              continue
   839
   840                          # Buy signal
   841                          if (position == 0 and
   842                              prob > params['BUY_PROB_THRESHOLD'] and
   843                              confidence > params['CONFIDENCE_THRESHOLD']):
   844
   845                              buy_fee = capital * params['TAKER_FEE_RATE']
   846                              btc_amount = (capital - buy_fee) / price
   847                              entry_price = price
   848                              highest_price_since_entry = price
   849                              position = 1
   850                              take_profit = entry_price * (1 + params['TAKE_PROFIT_PCT'])
   851                              stop_loss = entry_price * (1 - params['STOP_LOSS_PCT'])
   852                              open_trade_levels = {
   853                                  'take_profit': take_profit,
   854                                  'stop_loss': stop_loss,
   855                                  'buy_fee': buy_fee,
   856                                  'entry_confidence': confidence
   857                              }
   858                              self._candles_in_position = 0
   859                              self._low_prob_count = 0
   860
   861                              trade_entry_timestamp = timestamp
   862                              buy_probability = prob
   863
   864                          # Sell signal   865                          elif position == 1:
   866                              if price > highest_price_since_entry:
   867                                  highest_price_since_entry = price
   868
   869                              stop_loss = entry_price * (1 - params['STOP_LOSS_PCT'])
   870                              take_profit = entry_price * (1 + params['TAKE_PROFIT_PCT'])
   871                              trailing_stop = highest_price_since_entry * (1 - params.get('TRAIL_STOP_PCT', 0.02))
   872
   873                              candles_in_position = getattr(self, '_candles_in_position', 0) + 1
   874                              self._candles_in_position = candles_in_position
   875
   876                              if not hasattr(self, '_low_prob_count'):
   877                                  self._low_prob_count = 0
   878
   879                              if prob < 0.3:
   880                                  self._low_prob_count += 1
   881                              else:
   882                                  self._low_prob_count = 0
   883
   884                              exit_reason = None
   885
   886                              # Check low price for stop-loss and high price for take-profit
   887                              if float(candle['Low']) <= stop_loss:
   888                                  exit_reason = 'Stop Loss'
   889                                  exit_price = stop_loss
   890                              elif float(candle['High']) >= take_profit:
   891                                  exit_reason = 'Take Profit'
   892                                  exit_price = take_profit
   893                              elif price <= trailing_stop and price >= entry_price * 1.001:
   894                                  exit_reason = 'Trailing Stop'
   895                                  exit_price = price
   896                              elif (self._low_prob_count >= 3 and
   897                                    candles_in_position > 6 and
   898                                    prob < 0.25):
   899                                  exit_reason = 'Signal Exit'
   900                                  exit_price = price
   901
   902                              if exit_reason:
   903                                  sell_fee = btc_amount * exit_price * params['TAKER_FEE_RATE']
   904                                  total_trade_fees = open_trade_levels['buy_fee'] + sell_fee
   905                                  total_fees += total_trade_fees
   906
   907                                  raw_pnl = (exit_price - entry_price) * btc_amount
   908                                  pnl = raw_pnl - total_trade_fees
   909
   910                                  capital += pnl + total_trade_fees
   911
   912                                  duration = timestamp - trade_entry_timestamp if trade_entry_timestamp else pd.Timedelta(0)
   913
   914                                  trades.append({
   915                                      'timestamp': str(timestamp),
   916                                      'entry_price': entry_price,
   917                                      'exit_price': exit_price,
   918                                      'exit_reason': exit_reason,
   919                                      'pnl': pnl,
   920                                      'raw_pnl': raw_pnl,
   921                                      'buy_probability': buy_probability,
   922                                      'confidence': confidence,
   923                                      'entry_confidence': open_trade_levels.get('entry_confidence', 0),
   924                                      'fees': total_trade_fees,
   925                                      'highest_price': highest_price_since_entry,
   926                                      'hold_duration': candles_in_position,
   927                                      'entry_date': str(trade_entry_timestamp) if trade_entry_timestamp else str(timestamp),
   928                                      'exit_date': str(timestamp),
   929                                      'status': 'CLOSED',
   930                                      'stop_loss': stop_loss,
   931                                      'take_profit': take_profit,
   932                                      'position_size': btc_amount,
   933                                      'trade_usd_amount': params['START_CAPITAL'] - open_trade_levels['buy_fee'],
   934                                      'duration': str(duration)
   935                                  })
   936
   937                                  trade_type = "WIN" if pnl > 0 else "LOSS"
   938                                  self._log(f" ⭐ {trade_type}: ${entry_price:.2f} → ${exit_price:.2f} | P&L: ${raw_pnl:.2f} | P&L (net): ${pnl:.2f} | Fees: ${total_trade_fees:.2f}")
   939                                  position = 0
   940                                  self._candles_in_position = 0
   941                                  self._low_prob_count = 0
   942                                  open_trade_levels = {}
   943                                  self.results['trades'] = trades
   944                                  trade_entry_timestamp = None
   945                                  buy_probability = 0.0
   946
   947                          total_processed += 1
   948
   949                      except Exception as e:
   950                          logger.error(f"Error processing candle {i}: {e}")
   951                          continue
   952
   953                  progress = (chunk_end / len(data)) * 100
   954                  self._log(f"📈 Processing... {progress:.1f}% complete ({len(trades)} trades so far)")
   955
   956              if trades:
   957                  self._calculate_metrics(trades, params['START_CAPITAL'], total_fees)
   958              else:
   959                  self._log("⚠️ No trades executed during backtest")
   960                  self.results['metrics'] = {
   961                      'total_trades': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_win': 0,
   962                      'avg_loss': 0, 'return_pct': 0, 'total_fees': 0
   963                  }
   964
   965              final_capital = params['START_CAPITAL'] + sum(t['pnl'] for t in trades)
   966              self._log(f"💸 Final Capital: ${final_capital:.2f}")
   967
   968              self.results['trades'] = trades
   969              self.results['equity_curve'] = equity_curve
   970
   971              self._log(f"✅ Backtest completed - {len(trades)} trades executed")
   972
   973          except Exception as e:
   974              self._log(f"❌ Backtest error: {str(e)}")
   975              logger.error(f"Backtest error: {e}")
   976              traceback.print_exc()
   977          finally:
   978              self.running = False
   979              self.results['running'] = False
   980
   981      def _get_backtest_data(self, symbol: str, interval: str, start_date: str, end_date: str, period: str = '30d') -> pd.DataFrame:
   982          """Get data for backtesting"""
   983          has_dates = start_date and end_date and start_date.strip() and end_date.strip()
   984
   985          if has_dates:
   986              try:
   987                  start_dt = datetime.strptime(start_date, '%Y-%m-%d')
   988                  end_dt = datetime.strptime(end_date, '%Y-%m-%d')
   989                  days_diff = (end_dt - start_dt).days
   990                  self._log(f"📅 Using specific date range: {start_date} to {end_date} ({days_diff} days)")
   991
   992                  data = DataProvider.get_binance_data(symbol, interval, start_date, end_date)
   993
   994                  if data.empty:
   995                      calculated_period = f"{max(days_diff, 7)}d"
   996                      self._log(f"🔄 Binance unavailable, using Yahoo Finance with {calculated_period}...")
   997                      yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
   998                      data = DataProvider.get_yfinance_data(yf_symbol, calculated_period, interval)
   999
  1000                      if not data.empty:
  1001                          try:
  1002                              start_dt_pd = pd.to_datetime(start_date)
  1003                              end_dt_pd = pd.to_datetime(end_date) + pd.Timedelta(days=1)
  1004                              data = data[(data.index >= start_dt_pd) & (data.index < end_dt_pd)]
  1005                              self._log(f"🤔 Filtered to exact date range: {len(data)} candles")
  1006                          except Exception as e:
  1007                              self._log(f"⚠️ Date filtering failed: {e}")
  1008                  return data
  1009
  1010              except Exception as e:
  1011                  self._log(f"📅 Date parsing error: {e}, falling back to period: {period}")
  1012
  1013          self._log(f"📅 Using backtest period: {period} (no specific dates)")
  1014          data = DataProvider.get_backtest_data_with_period(symbol, period, interval)
  1015
  1016          return data
  1017
  1018      def _calculate_metrics(self, trades: List[Dict], start_capital: float, total_fees: float):
  1019          """Calculate backtest performance metrics"""
  1020          try:
  1021              if not trades:
  1022                  self.results['metrics'] = {}
  1023                  return
  1024
  1025              winning_trades = [t for t in trades if t['pnl'] > 0]
  1026              losing_trades = [t for t in trades if t['pnl'] <= 0]
  1027
  1028              winner_amount = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
  1029              loser_amount = sum([t['pnl'] for t in losing_trades]) if losing_trades else 0
  1030              total_pnl_with_fees = winner_amount + loser_amount
  1031
  1032              win_rate = len(winning_trades) / len(trades) if trades else 0
  1033              avg_win = winner_amount / len(winning_trades) if winning_trades else 0  1034              avg_loss = abs(loser_amount) / len(losing_trades) if losing_trades else 0
  1035              win_loss_ratio = len(winning_trades) / len(losing_trades) if losing_trades else len(winning_trades)
  1036
  1037              return_pct = (total_pnl_with_fees / start_capital) * 100
  1038
  1039              metrics = {
  1040                  'total_trades': len(trades),
  1041                  'winner_count': len(winning_trades),
  1042                  'winner_amount': winner_amount,
  1043                  'loser_count': len(losing_trades),
  1044                  'loser_amount': abs(loser_amount),
  1045                  'total_pnl': total_pnl_with_fees,
  1046                  'win_rate': win_rate,  1047                  'avg_win': avg_win,
  1048                  'avg_loss': avg_loss,  1049                  'win_loss_ratio': win_loss_ratio,
  1050                  'return_pct': return_pct,
  1051                  'total_fees': total_fees
  1052              }
  1053
  1054              self.results['metrics'] = metrics
  1055              self._log(f"💰 === P&L INCLUDING FEES ===")
  1056              self._log(f"💰 Total P&L (with fees): ${total_pnl_with_fees:.2f}")
  1057              self._log(f"💰 Total Fees Paid: ${total_fees:.2f}")
  1058              self._log(f"📊 Final Results: {len(trades)} trades, {len(winning_trades)} winners (${winner_amount:.2f}), {len(losing_trades)} losers (${abs(loser_amount):.2f})")
  1059              self._log(f"📊 Win/Loss Ratio: {win_loss_ratio:.2f}")
  1060              self._log(f"📊 Total Return: {return_pct:.2f}%")
  1061
  1062          except Exception as e:
  1063              logger.error(f"Metrics calculation error: {e}")
  1064              self.results['metrics'] = {}
  1065
  1066      def _log(self, message: str):
  1067          """Add log message"""
  1068          timestamp = datetime.now().strftime('%H:%M:%S')
  1069          log_entry = f"{timestamp} | {message}"
  1070          self.logs.append(log_entry)
  1071          if len(self.logs) > 50:
  1072              self.logs = self.logs[-50:]
  1073          logger.info(message)
  1074
  1075      def stop(self):
  1076          """Stop backtest"""
  1077          self.running = False
  1078          self._log("🛑 Backtest stopped by user")
  1079
  1080  class LiveTrader:
  1081      """Live trader with single active trade and detailed logging, matching backtester logic."""
  1082
  1083      def __init__(self):
  1084          self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}
  1085          self.logs = []
  1086          self.running = False
  1087          self.model = None
  1088          self.params = {}
  1089          self.data_history = pd.DataFrame()
  1090          self._last_candle_timestamp = None
  1091
  1092      def start(self, params, model):
  1093          full_params = deepcopy(TradingConfig.DEFAULT_PARAMS)
  1094          full_params.update(params)
  1095          self.params = full_params
  1096          self.running = True
  1097          self.model = model
  1098          self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': True}
  1099          self.logs = []
  1100          self._last_candle_timestamp = None
  1101
  1102          try:
  1103              script_filepath = os.path.abspath(__file__)
  1104              self._log(f"✅ Script Filepath: {script_filepath}")
  1105          except NameError:
  1106              self._log("⚠️ Could not determine script path.")
  1107
  1108          self._log(f"🚀 Live trading started with ${self.params.get('START_CAPITAL', 10000):.2f}")
  1109          self._log(f"📊 Model: {model.model_type}, Accuracy: {getattr(model, 'training_accuracy', 0.0):.2%}")
  1110          self._log(f"📍 Symbol: {self.params.get('SYMBOL', 'Unknown')} | Interval: {self.params.get('INTERVAL', 'Unknown')}")
  1111          self._log(self._get_settings_log())
  1112
  1113          # Initial data fetch and feature extraction
  1114          initial_data = DataProvider.get_live_data(self.params['SYMBOL'], self.params['INTERVAL'])
  1115
  1116          # NOTE: Using the entire initial data set for feature extraction
  1117          self.data_history = TechnicalAnalysis.extract_features(
  1118              initial_data, self.params.get('FUTURE_WINDOW', 1), is_live_trading=True
  1119          )
  1120
  1121          if self.data_history.empty:
  1122              self._log("❌ Initial data fetch failed or insufficient data after feature extraction. Cannot start live trading.")  1123              self.running = False
  1124              return
  1125
  1126          self._last_candle_timestamp = self.data_history.index[-1]
  1127          self._log(f"✅ Initial data history loaded up to: {self._last_candle_timestamp}")
  1128
  1129          threading.Thread(target=self._live_trading_loop, daemon=True).start()
  1130
  1131      def stop(self):
  1132          self.running = False
  1133          self._log("🛑 Live trading stopped")
  1134
  1135      def _get_and_process_data(self) -> pd.DataFrame:
  1136          """Holt die neuesten Daten und extrahiert Features."""
  1137          data = DataProvider.get_live_data(self.params['SYMBOL'], self.params['INTERVAL'])
  1138          if data.empty or data.index.empty:
  1139              return pd.DataFrame()
  1140
  1141          featured_data = TechnicalAnalysis.extract_features(
  1142              data, self.params.get('FUTURE_WINDOW', 1), is_live_trading=True
  1143          )
  1144          return featured_data
  1145
  1146
  1147      # --- FIX: CORRECTED LIVE TRADING LOOP FOR DELAY ISSUE ---
  1148      def _live_trading_loop(self):
  1149          try:
  1150              interval_str = self.params.get('INTERVAL', '5m')
  1151              # Extract interval in minutes (works for '5m', '15m', '1h', etc.)
  1152              if 'm' in interval_str:
  1153                  interval_minutes = int(interval_str.rstrip('m'))
  1154              elif 'h' in interval_str:  1155                  interval_minutes = int(interval_str.rstrip('h')) * 60
  1156              else:
  1157                  interval_minutes = 5 # Default to 5 minutes
  1158
  1159              while self.running:
  1160                  # 1. Calculate time until the next expected candle CLOSE time
  1161                  now = datetime.now()
  1162                  # Find the next minute that is a multiple of the interval_minutes past the hour
  1163                  current_minute = now.minute
  1164
  1165                  # Minutes until the next interval boundary
  1166                  minutes_to_wait = (interval_minutes - (current_minute % interval_minutes)) % interval_minutes
  1167
  1168                  # Seconds remaining in the current minute (plus the full minutes to wait)
  1169                  seconds_to_wait = (minutes_to_wait * 60) + (60 - now.second)
  1170
  1171                  # We wait until *after* the candle close, plus a small API buffer (3s)
  1172                  wait_time_with_buffer = seconds_to_wait + 3
  1173
  1174                  # If we calculated a very small wait time (meaning we just missed the candle close),
  1175                  # we wait for the *next* full interval instead.
  1176                  if wait_time_with_buffer < 15:
  1177                      wait_time_with_buffer += interval_minutes * 60
  1178
  1179                  self._log(f"💤 Waiting {wait_time_with_buffer:.1f}s for the next candle close at approx. { (now + timedelta(seconds=wait_time_with_buffer)).strftime('%H:%M:%S') }...")
  1180
  1181                  time.sleep(wait_time_with_buffer)
  1182
  1183                  # 2. Short-Retry-Loop: Poll quickly until the new candle appears
  1184                  retries = 0
  1185                  max_retries = 10 # 10 retries * 5s = 50s total buffer
  1186
  1187                  while self.running and retries < max_retries:
  1188                      featured_data = self._get_and_process_data()
  1189
  1190                      if featured_data.empty:
  1191                          self._log("  Data fetch/feature extraction failed. Retrying in 5s.")
  1192                          time.sleep(5)  1193                          retries += 1
  1194                          continue
  1195
  1196                      # Filter for candles that are truly new (index > last processed timestamp)
  1197                      new_candles = featured_data[featured_data.index > self._last_candle_timestamp]
  1198
  1199                      if not new_candles.empty:
  1200                          # Process only the very latest candle
  1201                          latest_candle = featured_data.iloc[-1]
  1202                          self._log(f"� Found {len(new_candles)} new candle(s) after {retries} retries.")
  1203
  1204                          # Process the latest candle for trading decisions
  1205                          self._process_single_candle(latest_candle)
  1206
  1207                          # Update history and last timestamp
  1208                          self._last_candle_timestamp = latest_candle.name
  1209                          self.data_history = featured_data.tail(50)
  1210                          break # Exit retry loop, return to main sleep
  1211                      else:
  1212                          # Candle not yet available (API delay)
  1213                          self._log(f"⚠ No new candle found since {self._last_candle_timestamp}. Retry {retries+1}/{max_retries} in 5s.")
  1214                          retries += 1
  1215                          time.sleep(5)  1216
  1217                  if retries >= max_retries:
  1218                      self._log("🚨 MAX RETRIES REACHED. Candle seriously delayed or API down. Resuming initial sleep cycle.")
  1219
  1220          except Exception as e:
  1221              self._log(f"❌ Live trading loop error: {e}")
  1222              self.running = False
  1223              traceback.print_exc()
  1224      # ----------------------------------------------------
  1225
  1226
  1227      def _process_single_candle(self, candle: pd.Series):
  1228          """Processes a single candle for live trading decisions"""
  1229          try:
  1230              price = float(candle['Close'])
  1231              timestamp = candle.name
  1232
  1233              capital = self.params.get('START_CAPITAL', 10000)
  1234
  1235              trades = self.results.get('trades', [])
  1236              equity_curve = self.results.get('equity_curve', [])
  1237              open_trade = next((t for t in trades if t.get('status') == 'OPEN'), None)
  1238
  1239              equity = capital if not open_trade else open_trade.get('capital_after_entry', 0) + open_trade.get('unrealized_pnl', 0)
  1240              equity_curve.append({'timestamp': str(timestamp), 'equity': equity})
  1241              self.results['equity_curve'] = equity_curve
  1242
  1243              # Add a check to ensure features are present
  1244              if not all(f in candle.index for f in self.model.features):
  1245                  self._log("❌ Missing features in candle. Cannot make a prediction.")
  1246                  return
  1247
  1248              features = candle[self.model.features].values
  1249              prob, confidence = self.model.predict(features, is_live_trading=True)
  1250
  1251              self._log(
  1252                  f"🔍 Candle {timestamp} | Price=${price:.2f} | BuyProb={prob:.3f} | Conf={confidence:.3f} | Position: {'Open' if open_trade else 'Cash'}"
  1253              )
  1254
  1255              stop_loss_pct = self.params.get('STOP_LOSS_PCT')
  1256              take_profit_pct = self.params.get('TAKE_PROFIT_PCT')
  1257              trail_stop_pct = self.params.get('TRAIL_STOP_PCT')
  1258              signal_exit_threshold = self.params.get('SIGNAL_EXIT_THRESHOLD')
  1259              signal_exit_consecutive = self.params.get('SIGNAL_EXIT_CONSECUTIVE')
  1260              min_hold_candles = self.params.get('MIN_HOLD_CANDLES')
  1261              buy_prob_threshold = self.params.get('BUY_PROB_THRESHOLD')
  1262              confidence_threshold = self.params.get('CONFIDENCE_THRESHOLD')
  1263              taker_fee_rate = self.params.get('TAKER_FEE_RATE')
  1264
  1265              # BUY
  1266              if not open_trade and prob > buy_prob_threshold and confidence > confidence_threshold:
  1267                  position_size_usd = capital * 0.99
  1268                  fee = position_size_usd * taker_fee_rate
  1269                  btc_amount = (position_size_usd - fee) / price
  1270                  entry_price = price
  1271                  highest_price_since_entry = price
  1272                  capital_after_entry = capital - position_size_usd
  1273
  1274                  new_trade = {
  1275                      'trade_number': len(trades) + 1,
  1276                      'entry_timestamp': str(timestamp),
  1277                      'entry_price': entry_price,
  1278                      'position_size': btc_amount,
  1279                      'trade_usd_amount': position_size_usd, # New field for USDT amount
  1280                      'status': 'OPEN',  1281                      'capital_after_entry': capital_after_entry,
  1282                      'highest_price': highest_price_since_entry,
  1283                      'hold_duration': 1,
  1284                      'low_prob_count': 0,
  1285                      'total_fees': fee,
  1286                      'current_price': price,
  1287                      'unrealized_pnl': 0.0,
  1288                      'stop_loss': entry_price * (1 - stop_loss_pct),
  1289                      'take_profit': entry_price * (1 + take_profit_pct),
  1290                      'signal_exit_threshold': signal_exit_threshold,
  1291                      'signal_exit_consecutive': signal_exit_consecutive,
  1292                      'min_hold_candles': min_hold_candles,
  1293                      'symbol': self.params.get('SYMBOL', ''),
  1294                      'interval': self.params.get('INTERVAL', ''),
  1295                      'model_type': self.model.model_type
  1296                  }
  1297                  trades.append(new_trade)
  1298                  self.results['trades'] = trades
  1299                  self._log(f"🚀 Trade #{len(trades)} BUY TRIGGERED: Entry=${entry_price:.2f}")
  1300                  send_trade_email(new_trade)
  1301
  1302              # ON HOLD / SELL
  1303              elif open_trade:
  1304                  highest_price = max(open_trade['highest_price'], price)
  1305                  hold_duration = open_trade['hold_duration'] + 1
  1306                  low_prob_count = open_trade['low_prob_count'] + 1 if prob < 0.3 else 0
  1307
  1308                  unrealized_pnl = (price - open_trade['entry_price']) * open_trade['position_size']
  1309
  1310                  open_trade.update({
  1311                      'current_price': price,
  1312                      'highest_price': highest_price,
  1313                      'hold_duration': hold_duration,
  1314                      'low_prob_count': low_prob_count,
  1315                      'unrealized_pnl': unrealized_pnl,
  1316                  })
  1317
  1318                  stop_loss = open_trade['entry_price'] * (1 - stop_loss_pct)
  1319                  take_profit = open_trade['entry_price'] * (1 + take_profit_pct)
  1320                  trailing_stop = highest_price * (1 - trail_stop_pct)
  1321
  1322                  exit_reason = None
  1323                  exit_price = price
  1324
  1325                  # Check low price for stop-loss and high price for take-profit
  1326                  if float(candle['Low']) <= stop_loss:
  1327                      exit_reason = 'Stop Loss'
  1328                      exit_price = stop_loss
  1329                  elif float(candle['High']) >= take_profit:
  1330                      exit_reason = 'Take Profit'
  1331                      exit_price = take_profit
  1332                  elif price <= trailing_stop and price >= open_trade['entry_price'] * 1.001:
  1333                      exit_reason = 'Trailing Stop'
  1334                      exit_price = price
  1335                  elif (low_prob_count >= signal_exit_consecutive and
  1336                        hold_duration > min_hold_candles and
  1337                        prob < signal_exit_threshold):
  1338                      exit_reason = 'Signal Exit'
  1339                      exit_price = price
  1340
  1341                  if exit_reason:
  1342                      exit_fee = open_trade['position_size'] * exit_price * taker_fee_rate
  1343                      total_fees = open_trade['total_fees'] + exit_fee
  1344
  1345                      raw_pnl = (exit_price - open_trade['entry_price']) * open_trade['position_size']
  1346                      pnl = raw_pnl - total_fees
  1347
  1348                      open_trade.update({
  1349                          'exit_price': exit_price,
  1350                          'exit_reason': exit_reason,
  1351                          'status': 'CLOSED',
  1352                          'raw_pnl': raw_pnl,
  1353                          'pnl': pnl,
  1354                          'fees': total_fees,
  1355                          'unrealized_pnl': 0.0,
  1356                          'closed_timestamp': str(timestamp)
  1357                      })
  1358
  1359                      self._log(f"🎯 Trade #{open_trade['trade_number']} SELL ({exit_reason}): Exit=${exit_price:.2f}, P&L=${pnl:.2f}")
  1360                      send_trade_email(open_trade)
  1361
  1362              log_data = {
  1363                  'price': f"${price:.2f}",
  1364                  'Next TP price': f"${open_trade['take_profit']:.2f}" if open_trade else "N/A",
  1365                  'Next SL price': f"${open_trade['stop_loss']:.2f}" if open_trade else "N/A",
  1366                  'Next Signal exit prob': f"{signal_exit_threshold:.2%}",
  1367                  'Next Trailstop price': f"${trailing_stop:.2f}" if open_trade and 'trailing_stop' in locals() else "N/A",
  1368                  'conf': f"{confidence:.2%}",
  1369                  'Buy proba': f"{prob:.2%}"
  1370              }
  1371              self._log(f"📊 DATA: {log_data}")
  1372
  1373          except Exception as e:
  1374              self._log(f"❌ Live simulation error: {str(e)}")
  1375              traceback.print_exc()
  1376
  1377      def manual_close_trade(self) -> Tuple[bool, str]:
  1378          """Manually close an open trade at the current market price."""
  1379          try:
  1380              open_trade = next((t for t in self.results['trades'] if t.get('status') == 'OPEN'), None)
  1381
  1382              if not open_trade:
  1383                  self._log("MANUAL CLOSE: No open trade to close.")
  1384                  return False, "No open trade to close."
  1385
  1386              if self.data_history.empty:
  1387                  self._log("MANUAL CLOSE: Cannot close, no recent price data available.")
  1388                  return False, "No recent price data available."
  1389
  1390              exit_price = self.data_history['Close'].iloc[-1]
  1391              timestamp = self.data_history.index[-1]
  1392              taker_fee_rate = self.params.get('TAKER_FEE_RATE')
  1393
  1394              exit_fee = open_trade['position_size'] * exit_price * taker_fee_rate
  1395              total_fees = open_trade['total_fees'] + exit_fee
  1396              raw_pnl = (exit_price - open_trade['entry_price']) * open_trade['position_size']
  1397              pnl = raw_pnl - total_fees
  1398
  1399              open_trade.update({
  1400                  'exit_price': exit_price,
  1401                  'exit_reason': 'Manual Close',
  1402                  'status': 'CLOSED',
  1403                  'raw_pnl': raw_pnl,
  1404                  'pnl': pnl,
  1405                  'fees': total_fees,
  1406                  'unrealized_pnl': 0.0,
  1407                  'closed_timestamp': str(timestamp)
  1408              })
  1409
  1410              self._log(f"✅ MANUAL CLOSE: Trade #{open_trade['trade_number']} closed at ${exit_price:.2f}, P&L=${pnl:.2f}")
  1411              send_trade_email(open_trade)
  1412              return True, f"Trade manually closed at ${exit_price:.2f}."
  1413
  1414          except Exception as e:
  1415              self._log(f"❌ MANUAL CLOSE ERROR: {str(e)}")
  1416              traceback.print_exc()
  1417              return False, f"An error occurred: {str(e)}"
  1418
  1419      def _get_settings_log(self):
  1420          defaults = deepcopy(TradingConfig.DEFAULT_PARAMS)
  1421          for k, v in self.params.items():
  1422              defaults[k] = v
  1423
  1424          p = defaults
  1425          return (
  1426              f"Settings: Symbol={p.get('SYMBOL')} | Interval={p.get('INTERVAL')} | "
  1427              f"Capital={p.get('START_CAPITAL')} | SL={p.get('STOP_LOSS_PCT')} | TP={p.get('TAKE_PROFIT_PCT')} | "
  1428              f"TRL={p.get('TRAIL_STOP_PCT')} | BuyProbThresh={p.get('BUY_PROB_THRESHOLD')} | "
  1429              f"ModelType={getattr(self.model, 'model_type', 'N/A')} | ConfidenceThresh={p.get('CONFIDENCE_THRESHOLD')} | "
  1430              f"SignalExitThresh={p.get('SIGNAL_EXIT_THRESHOLD')} | SignalExitConsec={p.get('SIGNAL_EXIT_CONSECUTIVE')} | MinHoldCandles={p.get('MIN_HOLD_CANDLES')}"
  1431          )
  1432
  1433      def _log(self, message: str):
  1434          """Adds log message to both in-memory list and file log"""
  1435          timestamp = datetime.now().strftime('%H:%M:%S')
  1436          log_entry = f"{timestamp} | {message}"
  1437          self.logs.append(log_entry)
  1438          if len(self.logs) > 50:
  1439              self.logs = self.logs[-50:]
  1440
  1441          logger.info(message)
  1442
  1443      def get_status(self):
  1444          open_trade = next((t for t in self.results['trades'] if t.get('status') == 'OPEN'), None)
  1445          closed_trades = [t for t in self.results['trades'] if t.get('status') == 'CLOSED']
  1446          latest_price = self.data_history['Close'].iloc[-1] if not self.data_history.empty and 'Close' in self.data_history.columns else 0
  1447          total_trades = len(self.results['trades'])
  1448          pnl = 0.0
  1449          capital = self.params.get('START_CAPITAL', 10000)
  1450          position = 'Cash'
  1451
  1452          if open_trade:
  1453              unrealized_pnl = (latest_price - open_trade['entry_price']) * open_trade['position_size']
  1454              open_trade['unrealized_pnl'] = unrealized_pnl
  1455
  1456              pnl = unrealized_pnl
  1457              position = 'Open'
  1458              capital = open_trade.get('capital_after_entry', capital) + pnl
  1459          elif closed_trades:
  1460              pnl = closed_trades[-1]['pnl']
  1461              position = 'Cash'
  1462              capital = self.get_final_capital()
  1463
  1464          return {
  1465              'running': self.running,
  1466              'status': 'Running - Live Trading' if self.running else 'Stopped',
  1467              'current_price': latest_price,
  1468              'position': position,
  1469              'pnl': pnl,
  1470              'total_trades': total_trades,
  1471              'capital': capital,
  1472              'trade_details': open_trade if open_trade else (closed_trades[-1] if closed_trades else None)
  1473          }
  1474
  1475      def get_final_capital(self):
  1476          final_capital = self.params.get('START_CAPITAL', 10000)
  1477          closed_pnl = sum(t.get('pnl', 0) for t in self.results['trades'] if t.get('status') == 'CLOSED')
  1478          final_capital += closed_pnl
  1479
  1480          open_trade = next((t for t in self.results['trades'] if t.get('status') == 'OPEN'), None)
  1481          if open_trade:
  1482              final_capital += open_trade.get('unrealized_pnl', 0)
  1483
  1484          return final_capital
  1485
  1486      def get_live_data_for_chart(self):
  1487          try:
  1488              if self.data_history.empty:
  1489                  return {'priceData': [], 'trades': [], 'openTrade': None}
  1490
  1491              price_data = [
  1492                  {'x': str(ts), 'y': price}
  1493                  for ts, price in self.data_history['Close'].items()
  1494              ]
  1495
  1496              closed_trades = [
  1497                  {
  1498                      'status': 'CLOSED',
  1499                      'entry_price': t['entry_price'],
  1500                      'exit_price': t['exit_price'],
  1501                      'entry_date': str(pd.to_datetime(t['entry_timestamp'])),
  1502                      'exit_date': str(pd.to_datetime(t['closed_timestamp'])),
  1503                      'pnl': t['pnl']
  1504                  }
  1505                  for t in self.results['trades'] if t.get('status') == 'CLOSED'
  1506              ]
  1507
  1508              open_trade = next((t for t in self.results['trades'] if t.get('status') == 'OPEN'), None)
  1509              open_trade_data = None
  1510              if open_trade:
  1511                  open_trade_data = {
  1512                      'entry_price': open_trade['entry_price'],
  1513                      'entry_date': str(pd.to_datetime(open_trade['entry_timestamp'])),
  1514                      'stop_loss': open_trade['stop_loss'],
  1515                      'take_profit': open_trade['take_profit'],
  1516                      'trail_stop_pct': open_trade.get('trail_stop_pct', self.params.get('TRAIL_STOP_PCT')), # Ensure it exists
  1517                      'highest_price': open_trade['highest_price']
  1518                  }
  1519
  1520              return {
  1521                  'priceData': price_data,
  1522                  'closedTrades': closed_trades,
  1523                  'openTrade': open_trade_data
  1524              }
  1525          except Exception as e:
  1526              logger.error(f"Error preparing chart data: {e}")
  1527              return {'priceData': [], 'closedTrades': [], 'openTrade': None}
  1528
  1529  # --- Flask App Routes ---
  1530  app = Flask(__name__)
  1531  backtest_engine = BacktestEngine()
  1532  config = TradingConfig.DEFAULT_PARAMS.copy()
  1533  settings_manager = SettingsManager()
  1534
  1535  live_trader = None
  1536  live_trading_running = False
  1537  latest_backtest_settings = None
  1538  current_trained_model = None
  1539  latest_model_info = {
  1540      'training_accuracy': 0.0, 'test_accuracy': 0.0, 'validation_accuracy': 0.0, 'hit_rate': 0.0,
  1541      'model_type': 'Not Trained', 'features_count': 0, 'class_distribution': {}
  1542  }
  1543
  1544  # --- MODIFIED: Dynamic filename and folder display ---
  1545  @app.route('/')
  1546  def dashboard():
  1547      script_path = os.path.abspath(sys.argv[0])
  1548      script_filename = os.path.basename(script_path)
  1549      script_folder = os.path.basename(os.path.dirname(script_path))
  1550
  1551      return render_template('dashboard.html',
  1552                             script_filename=script_filename,
  1553                             script_folder=script_folder)
  1554
  1555  # --- Original routes ---
  1556  @app.route('/backtest')
  1557  def backtest_page():
  1558      return render_template('backtest.html')
  1559
  1560  @app.route('/status')
  1561  def get_status():
  1562      global latest_model_info
  1563      return jsonify({
  1564          'running': False, 'capital': config.get('START_CAPITAL', 10000), 'current_price': 0, 'position': 0,
  1565          'pnl': 0, 'total_trades': 0, 'model_confidence': 0,
  1566          'model_type': latest_model_info.get('model_type', 'Not Trained'),
  1567          'last_trade': 'None',
  1568          'training_accuracy': latest_model_info.get('training_accuracy', 0.0),
  1569          'test_accuracy': latest_model_info.get('test_accuracy', 0.0),
  1570          'validation_accuracy': latest_model_info.get('validation_accuracy', 0.0),
  1571          'hit_rate': latest_model_info.get('hit_rate', 0.0),
  1572          'features_count': latest_model_info.get('features_count', 0),
  1573          'signal_exit_threshold': config.get('SIGNAL_EXIT_THRESHOLD', 0.25),
  1574          'signal_exit_consecutive': config.get('SIGNAL_EXIT_CONSECUTIVE', 3),
  1575          'min_hold_candles': config.get('MIN_HOLD_CANDLES', 6),
  1576          'total_return_pct': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0
  1577      })
  1578
  1579  @app.route('/params')
  1580  def get_params():
  1581      return jsonify(config)
  1582
  1583  @app.route('/update_params', methods=['POST'])
  1584  def update_params():
  1585      global config
  1586      try:
  1587          data = request.json
  1588          for key, value in data.items():
  1589              if key in config:
  1590                  config[key] = value
  1591          return jsonify({'status': 'Parameters updated'})
  1592      except Exception as e:
  1593          return jsonify({'status': f'Error: {str(e)}'})
  1594
  1595  @app.route('/trades')
  1596  def get_trades():
  1597      return jsonify([])
  1598
  1599  @app.route('/start_backtest', methods=['POST'])
  1600  def start_backtest():
  1601      global latest_backtest_settings
  1602      try:
  1603          data = request.json
  1604          params = data.get('params', config)
  1605          start_date = data.get('start_date', '')
  1606          end_date = data.get('end_date', '')
  1607
  1608          latest_backtest_settings = params.copy()
  1609
  1610          if backtest_engine.running:
  1611              return jsonify({'status': 'Backtest already running'})
  1612
  1613          def run_backtest():
  1614              backtest_engine.run(params, start_date, end_date)
  1615
  1616          threading.Thread(target=run_backtest, daemon=True).start()
  1617          return jsonify({'status': 'Backtest started'})
  1618
  1619      except Exception as e:
  1620          logger.error(f"Start backtest error: {e}")
  1621          return jsonify({'status': f'Error: {str(e)}'})
  1622
  1623  @app.route('/stop_backtest', methods=['POST'])
  1624  def stop_backtest():
  1625      try:
  1626          backtest_engine.stop()
  1627          return jsonify({'status': 'Backtest stopped'})
  1628      except Exception as e:
  1629          return jsonify({'status': f'Error: {str(e)}'})
  1630
  1631  @app.route('/backtest_results')
  1632  def get_backtest_results():
  1633      try:
  1634          return jsonify(backtest_engine.results)
  1635      except Exception as e:
  1636          logger.error(f"Get results error: {e}")
  1637          return jsonify({'error': str(e)})
  1638
  1639  @app.route('/backtest_logs')
  1640  def get_backtest_logs():
  1641      try:
  1642          return jsonify({'logs': backtest_engine.logs})
  1643      except Exception as e:
  1644          logger.error(f"Get logs error: {e}")
  1645          return jsonify({'logs': [f"Error getting logs: {str(e)}"]})
  1646
  1647  @app.route('/reset_backtest', methods=['POST'])
  1648  def reset_backtest():
  1649      try:
  1650          backtest_engine.stop()
  1651          backtest_engine.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}
  1652          backtest_engine.logs = ['Ready to start backtest...']
  1653          backtest_engine.running = False
  1654          logger.info("Backtest reset successfully")
  1655          return jsonify({'status': 'Backtest reset successfully'})
  1656      except Exception as e:
  1657          logger.error(f"Reset backtest error: {e}")
  1658          return jsonify({'status': f'Error: {str(e)}'})
  1659
  1660  @app.route('/save_model', methods=['POST'])
  1661  def save_current_model():
  1662      global current_trained_model
  1663      try:
  1664          if current_trained_model is None or not current_trained_model.is_trained:
  1665              return jsonify({'status': 'error', 'message': 'No trained model available to save'})
  1666          data = request.json or {}
  1667          custom_name = data.get('filename', None)
  1668          if custom_name:
  1669              if not custom_name.endswith('.pkl'):
  1670                  custom_name += '.pkl'  1671              filepath = f"models/{custom_name}"
  1672          else:
  1673              filepath = None
  1674          saved_path = current_trained_model.save_model(filepath)
  1675          return jsonify({
  1676              'status': 'success', 'message': f'Model saved successfully to {saved_path}', 'filepath': saved_path
  1677          })
  1678      except Exception as e:
  1679          logger.error(f"Save model error: {e}")
  1680          return jsonify({'status': 'error', 'message': str(e)})
  1681
  1682  @app.route('/load_model', methods=['POST'])
  1683  def load_saved_model():
  1684      global current_trained_model, latest_model_info
  1685      try:
  1686          data = request.json
  1687          filepath = data.get('filepath')
  1688          if not filepath:
  1689              return jsonify({'status': 'error', 'message': 'No filepath provided'})  1690          model = MLModel()
  1691          success = model.load_model(filepath)
  1692          if success:
  1693              current_trained_model = model
  1694              model_info = model.get_model_info()
  1695              latest_model_info.update({
  1696                  'training_accuracy': model_info.get('training_accuracy', 0.0), 'test_accuracy': model_info.get('test_accuracy', 0.0),
  1697                  'validation_accuracy': model_info.get('validation_accuracy', 0.0), 'hit_rate': model_info.get('hit_rate', 0.0),  1698                  'model_type': model_info.get('model_type', 'Unknown'), 'features_count': model_info.get('n_features', 0),
  1699                  'class_distribution': model_info.get('class_distribution', {})
  1700              })
  1701              return jsonify({
  1702                  'status': 'success', 'message': f'Model loaded successfully from {filepath}', 'model_info': model_info
  1703              })
  1704          else:
  1705              return jsonify({'status': 'error', 'message': 'Failed to load model'})  1706      except Exception as e:
  1707          logger.error(f"Load model error: {e}")
  1708          return jsonify({'status': 'error', 'message': str(e)})
  1709
  1710  @app.route('/list_models')
  1711  def list_saved_models():
  1712      try:
  1713          models = MLModel.list_saved_models()
  1714          return jsonify({'models': models})
  1715      except Exception as e:
  1716          logger.error(f"List models error: {e}")
  1717          return jsonify({'error': str(e)})
  1718
  1719  @app.route('/delete_model', methods=['POST'])
  1720  def delete_saved_model():
  1721      try:
  1722          data = request.json
  1723          filepath = data.get('filepath')
  1724          if not filepath or not os.path.exists(filepath):
  1725              return jsonify({'status': 'error', 'message': 'Model file not found'})  1726          os.remove(filepath)
  1727          return jsonify({'status': 'success', 'message': f'Model deleted: {filepath}'})
  1728      except Exception as e:
  1729          logger.error(f"Delete model error: {e}")
  1730          return jsonify({'status': 'error', 'message': str(e)})
  1731
  1732  @app.route('/save_settings', methods=['POST'])
  1733  def save_current_settings():
  1734      global config, settings_manager
  1735      try:
  1736          data = request.json or {}
  1737          custom_name = data.get('filename', None)
  1738          saved_path = settings_manager.save_settings(config, custom_name)
  1739          return jsonify({
  1740              'status': 'success', 'message': f'Settings saved successfully to {saved_path}', 'filepath': saved_path
  1741          })
  1742      except Exception as e:
  1743          logger.error(f"Save settings error: {e}")
  1744          return jsonify({'status': 'error', 'message': str(e)})
  1745
  1746  @app.route('/load_settings', methods=['POST'])
  1747  def load_saved_settings():
  1748      global config, settings_manager
  1749      try:
  1750          data = request.json
  1751          filepath = data.get('filepath')
  1752          if not filepath:
  1753              return jsonify({'status': 'error', 'message': 'No filepath provided'})  1754          loaded_settings = settings_manager.load_settings(filepath)
  1755          config.update(loaded_settings)
  1756          return jsonify({
  1757              'status': 'success', 'message': f'Settings loaded successfully from {filepath}', 'settings': loaded_settings
  1758          })
  1759      except Exception as e:
  1760          logger.error(f"Load settings error: {e}")
  1761          return jsonify({'status': 'error', 'message': str(e)})
  1762
  1763  @app.route('/list_settings')
  1764  def list_saved_settings():
  1765      try:
  1766          settings = settings_manager.list_saved_settings()
  1767          return jsonify({'settings': settings})
  1768      except Exception as e:
  1769          logger.error(f"List settings error: {e}")
  1770          return jsonify({'error': str(e)})
  1771
  1772  @app.route('/delete_settings', methods=['POST'])
  1773  def delete_saved_settings():
  1774      try:
  1775          data = request.json
  1776          filepath = data.get('filepath')
  1777          if not filepath:
  1778              return jsonify({'status': 'error', 'message': 'No filepath provided'})  1779          success = settings_manager.delete_settings(filepath)
  1780          if success:
  1781              return jsonify({'status': 'success', 'message': f'Settings deleted: {filepath}'})
  1782          else:
  1783              return jsonify({'status': 'error', 'message': 'Settings file not found'})
  1784      except Exception as e:
  1785          logger.error(f"Delete settings error: {e}")
  1786          return jsonify({'status': 'error', 'message': str(e)})
  1787
  1788  @app.route('/save_backtest_settings', methods=['POST'])
  1789  def save_backtest_settings():
  1790      global settings_manager
  1791      try:
  1792          data = request.json or {}
  1793          custom_name = data.get('filename', None)
  1794          params = data.get('params', {})
  1795          if not params:
  1796              return jsonify({'status': 'error', 'message': 'No parameters provided'})
  1797          saved_path = settings_manager.save_settings(params, custom_name)
  1798          return jsonify({
  1799              'status': 'success', 'message': f'Backtest settings saved successfully to {saved_path}', 'filepath': saved_path
  1800          })
  1801      except Exception as e:
  1802          logger.error(f"Save backtest settings error: {e}")
  1803          return jsonify({'status': 'error', 'message': str(e)})
  1804
  1805  @app.route('/create_project_zip', methods=['POST'])
  1806  def create_project_zip():
  1807      try:
  1808          import zipfile
  1809          from io import BytesIO
  1810          from flask import send_file
  1811          data = request.json or {}
  1812          custom_filename = data.get('filename', None)
  1813          if custom_filename:
  1814              if not custom_filename.endswith('.zip'):
  1815                  custom_filename += '.zip'
  1816              zip_filename = custom_filename
  1817          else:
  1818              timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  1819              zip_filename = f"bitcoin_trading_bot_project_{timestamp}.zip"
  1820          zip_buffer = BytesIO()
  1821          files_added = 0
  1822          exclude_dirs = {
  1823              '__pycache__', '.git', '.replit', 'node_modules', '.env', 'venv', '.venv', '.DS_Store', 'Thumbs.db', '.pytest_cache',
  1824              'attached_assets', 'bitcoin_bot_extracted'
  1825          }
  1826          exclude_files = {
  1827              'uv.lock', 'generated-icon.png', 'bitcoin_trading_bot_duplicate.zip', '.replit'
  1828          }
  1829          logger.info(f"🔧 Creating ZIP file: {zip_filename}")
  1830          with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
  1831              main_files = ['main.py', 'pyproject.toml', 'main5.py'] # Added main5.py for completeness
  1832              for main_file in main_files:
  1833                  if os.path.exists(main_file):
  1834                      zip_file.write(main_file, main_file)
  1835                      files_added += 1
  1836                      logger.info(f"✅ Added: {main_file}")
  1837              if os.path.exists('templates'):
  1838                  for root, dirs, files in os.walk('templates'):
  1839                      for file in files:
  1840                          if file.endswith(('.html', '.css', '.js')):
  1841                              file_path = os.path.join(root, file)
  1842                              archive_path = file_path.replace('\\', '/')
  1843                              zip_file.write(file_path, archive_path)
  1844                              files_added += 1
  1845                              logger.info(f"✅ Added: {archive_path}")
  1846              if os.path.exists('models'):
  1847                  for root, dirs, files in os.walk('models'):
  1848                      for file in files:
  1849                          if file.endswith('.pkl'):
  1850                              file_path = os.path.join(root, file)
  1851                              archive_path = file_path.replace('\\', '/')
  1852                              zip_file.write(file_path, archive_path)
  1853                              files_added += 1
  1854                              logger.info(f"✅ Added: {archive_path}")
  1855              if os.path.exists('settings'):
  1856                  for root, dirs, files in os.walk('settings'):
  1857                      for file in files:
  1858                          if file.endswith('.json'):
  1859                              file_path = os.path.join(root, file)
  1860                              archive_path = file_path.replace('\\', '/')
  1861                              zip_file.write(file_path, archive_path)
  1862                              files_added += 1
  1863                              logger.info(f"✅ Added: {archive_path}")
  1864              for file in os.listdir('.'):
  1865                  if (file.endswith('.py') and file not in exclude_files and file != 'main.py' and file != 'main5.py' and os.path.isfile(file)):
  1866                      zip_file.write(file, file)
  1867                      files_added += 1
  1868                      logger.info(f"✅ Added: {file}")
  1869          zip_buffer.seek(0)
  1870          if files_added == 0:
  1871              logger.error("❌ No files were added to ZIP")
  1872              return jsonify({'status': 'error', 'message': 'No files found to zip'}), 400
  1873          logger.info(f"✅ ZIP created successfully: {zip_filename} ({files_added} files)")
  1874          return send_file(
  1875              zip_buffer, as_attachment=True, download_name=zip_filename, mimetype='application/zip'
  1876          )
  1877      except Exception as e:
  1878          logger.error(f"❌ ZIP creation error: {e}")
  1879          traceback.print_exc()
  1880          return jsonify({'status': 'error', 'message': str(e)}), 500
  1881
  1882  @app.route('/get_latest_settings')
  1883  def get_latest_settings():
  1884      global latest_backtest_settings
  1885      try:
  1886          if latest_backtest_settings:
  1887              return jsonify({ 'status': 'success', 'settings': latest_backtest_settings })
  1888          else:
  1889              return jsonify({ 'status': 'error', 'message': 'No backtest settings found. Please run a backtest first.' })
  1890      except Exception as e:
  1891          return jsonify({'status': 'error', 'message': str(e)})
  1892
  1893  @app.route('/start_live_trading', methods=['POST'])
  1894  def start_live_trading():
  1895      global live_trader, live_trading_running, current_trained_model
  1896      try:
  1897          data = request.json
  1898          params = data.get('params', {})
  1899          if live_trading_running:
  1900              return jsonify({'status': 'error', 'message': 'Live trading already running'})
  1901          if not current_trained_model or not current_trained_model.is_trained:
  1902              return jsonify({'status': 'error', 'message': 'No trained model available. Please run a backtest first.'})
  1903
  1904          live_trader = LiveTrader()
  1905          live_trader.start(params, current_trained_model)
  1906          live_trading_running = True
  1907          logger.info("Live trading started")
  1908          return jsonify({ 'status': 'success', 'message': 'Live trading started successfully' })
  1909      except Exception as e:
  1910          logger.error(f"Start live trading error: {e}")
  1911          return jsonify({'status': 'error', 'message': str(e)})
  1912
  1913  @app.route('/stop_live_trading', methods=['POST'])
  1914  def stop_live_trading():
  1915      global live_trader, live_trading_running
  1916      try:
  1917          if live_trader:
  1918              live_trader.stop()
  1919          live_trading_running = False
  1920          logger.info("Live trading stopped")
  1921          return jsonify({ 'status': 'success', 'message': 'Live trading stopped' })  1922      except Exception as e:
  1923          logger.error(f"Stop live trading error: {e}")
  1924          return jsonify({'status': 'error', 'message': str(e)})
  1925
  1926  @app.route('/manual_close_trade', methods=['POST'])
  1927  def manual_close_trade():
  1928      global live_trader, live_trading_running
  1929      try:
  1930          if live_trader and live_trading_running:
  1931              success, message = live_trader.manual_close_trade()
  1932              if success:
  1933                  return jsonify({'status': 'success', 'message': message})
  1934              else:
  1935                  return jsonify({'status': 'error', 'message': message})
  1936          else:
  1937              return jsonify({'status': 'error', 'message': 'Live trading is not active.'})
  1938      except Exception as e:
  1939          logger.error(f"Manual close error: {e}")
  1940          return jsonify({'status': 'error', 'message': str(e)})
  1941
  1942  @app.route('/reset_live_trading', methods=['POST'])
  1943  def reset_live_trading():
  1944      global live_trader, live_trading_running
  1945      try:
  1946          if live_trader and live_trading_running:
  1947              live_trader.stop()
  1948
  1949          live_trader = None
  1950          live_trading_running = False
  1951
  1952          logger.info("Live trading session reset successfully")
  1953
  1954          return jsonify({'status': 'success', 'message': 'Live trading session reset successfully.'})
  1955      except Exception as e:
  1956          logger.error(f"Reset live trading error: {e}")
  1957          return jsonify({'status': 'error', 'message': str(e)})
  1958
  1959  @app.route('/live_trading_status')
  1960  def get_live_trading_status():
  1961      global live_trader, live_trading_running
  1962      try:
  1963          if live_trader and live_trading_running:
  1964              status = live_trader.get_status()
  1965              status['trades'] = live_trader.results['trades']
  1966              return jsonify(status)
  1967          else:
  1968              return jsonify({
  1969                  'running': False, 'status': 'Stopped', 'current_price': 0, 'position': 'Cash', 'pnl': 0,
  1970                  'total_trades': 0, 'capital': 0, 'trades': []
  1971              })
  1972      except Exception as e:
  1973          logger.error(f"Get live trading status error: {e}")
  1974          return jsonify({'running': False, 'error': str(e)})
  1975
  1976  @app.route('/live_trading_logs')
  1977  def get_live_trading_logs():
  1978      global live_trader
  1979      try:
  1980          if live_trader and hasattr(live_trader, 'logs'):
  1981              return jsonify({'logs': live_trader.logs})
  1982          else:
  1983              return jsonify({'logs': ['No live trading session active']})
  1984      except Exception as e:
  1985          logger.error(f"Get live trading logs error: {e}")
  1986          return jsonify({'logs': [f"Error getting logs: {str(e)}"]})
  1987
  1988  @app.route('/live_trading_chart_data')
  1989  def get_live_trading_chart_data():
  1990      global live_trader
  1991      if live_trader and live_trader.running:
  1992          data = live_trader.get_live_data_for_chart()
  1993          return jsonify(data)
  1994      else:
  1995          return jsonify({'priceData': [], 'closedTrades': [], 'openTrade': None})
  1996
  1997  @app.route('/ping')
  1998  def ping():
  1999      try:
  2000          return jsonify({
  2001              'status': 'alive', 'timestamp': datetime.now().isoformat(),
  2002              'server': 'Bitcoin Trading Bot', 'uptime': 'running'
  2003          })
  2004      except Exception as e:
  2005          return jsonify({'status': 'alive', 'error': str(e)})
  2006
  2007  if __name__ == '__main__':
  2008      logger.info("🚀 Starting Bitcoin Trading Bot Server")
  2009      app.run(host='0.0.0.0', port=5000, debug=True)
  2010

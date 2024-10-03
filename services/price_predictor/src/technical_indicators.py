import numpy as np
import pandas as pd
import talib

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe

    Args:
        df (pd.DataFrame): The input dataframe with the original features

    Returns:
        pd.DataFrame: The dataframe with the technical indicators added.
    """

    # Ensure we have the necessary columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # add the temporal features
    #df = add_temporal_features(df)
    
    # add the momentum indicators
    df = add_technical_indicators_momentum(df)
    
    # add the volatility indicators
    df = add_technical_indicators_volatility(df)
    
    # add the overlap indicators
    df = add_technical_indicators_overlap_studies(df)

    # add the volume indicators
    df = add_technical_indicators_volume(df)
    

    # # 1. Calculate a simple moving average (SMA): Measures the arithmetic mean of a given set of values over a specified period and gives equal weight to each value.
    # df['ma_7'] = talib.SMA(df['close'], timeperiod=7)
    # df['ma_14'] = talib.SMA(df['close'], timeperiod=14)
    # df['ma_28'] = talib.SMA(df['close'], timeperiod=28)

    # # 2. Calculate a exponential moving average (EMA): Similar to SMA but places more weight on recent data, making it more responsive to recent changes in price.
    # df['ema_7'] = talib.EMA(df['close'], timeperiod=7)
    # df['ema_14'] = talib.EMA(df['close'], timeperiod=14)
    # df['ema_28'] = talib.EMA(df['close'], timeperiod=28)

    # # 3. Relative Strength Index (RSI): Measures the speed and change of price movements, helping to identify overbought or oversold conditions
    # df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
    # df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    # df['rsi_28'] = talib.RSI(df['close'], timeperiod=28)

    # # 4. Moving Average Convergence Divergence (MACD): Shows the relationship between two moving averages of a price, useful for identifying trend direction and momentum.
    # df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

    # # 5. Bollinger Bands: Helps measure market volatility and identify overbought or oversold conditions.
    # df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(df['close'])

    # # 6. Stochastic Oscillator: Compares a closing price to its price range over a period of time, useful for identifying potential reversal points.
    # df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])

    # # 7. On-Balance Volume (OBV): Relates volume to price change, potentially predicting price movements based on volume trends.
    # df['obv'] = talib.OBV(df['close'], df['volume'])

    # # 10. Average True Range (ATR): Measures market volatility, which can be useful for setting stop-loss orders or identifying potential breakouts.
    # df['atr'] = talib.ATR(df['high'], df['low'], df['close'])

    # # 11. Commodity Channel Index (CCI): Helps identify cyclical trends and can signal overbought or oversold conditions.
    # df['cci_7'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=7)
    # df['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    # df['cci_28'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=28)

    # # 12.Money Flow Index (MFI): Combines price and volume data to measure buying and selling pressure.
    # df['mfi_7'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=7)
    # df['mfi_14'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    # df['mfi_28'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=28)

    # # 13. Chaikin Money Flow (CMF): Measures the buying and selling pressure over a specific period, useful for confirming price movements.
    # df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    # # 14. Rate of Change (ROC): Measures the percentage change in price over a specific period, helping to identify momentum.
    # df['roc_7'] = talib.ROC(df['close'], timeperiod=7)
    # df['roc_14'] = talib.ROC(df['close'], timeperiod=14)
    # df['roc_28'] = talib.ROC(df['close'], timeperiod=28)

    # # 15. Williams %R: Similar to the Stochastic Oscillator, it can help identify overbought and oversold levels.
    # df['willr_7'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=7)
    # df['willr_14'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    # df['willr_28'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=28)

    # # 16. Parabolic SAR: Useful for determining potential reversals in price direction.
    # df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)

    # # 17. Ichimoku Cloud (Conversion Line and Base Line): A comprehensive indicator that provides information about support, resistance, momentum, and trend direction.
    # # - ichimoku_conversion: Calculates the average of the 9-period high and 9-period low.
    # # - ichimoku_base: Calculates the average of the 26-period high and 26-period low.
    # #df['ichimoku_conversion'] = talib.TENKAN_SEN(df['high'], df['low'], timeperiod=9)
    # #df['ichimoku_base'] = talib.KIJUN_SEN(df['high'], df['low'], timeperiod=26)
    # df['ichimoku_conversion'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    # df['ichimoku_base'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2

    # # 18. Average Directional Index (ADX): Measures the strength of a trend
    # df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # # 19. Aroon Oscillator: Identifies the start of a new trend and its strength
    # df['aroon_osc'] = talib.AROONOSC(df['high'], df['low'], timeperiod=14)

    # # 20. Percentage Price Oscillator (PPO): Shows the relationship between two moving averages
    # df['ppo'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)

    # # 21. Keltner Channel: Similar to Bollinger Bands but uses ATR instead of standard deviation
    # def keltner_channel(high, low, close, period=20, atr_period=10, multiplier=2):
    #     typical_price = (high + low + close) / 3
    #     middle_line = talib.EMA(typical_price, timeperiod=period)
    #     atr = talib.ATR(high, low, close, timeperiod=atr_period)
    #     upper_line = middle_line + (multiplier * atr)
    #     lower_line = middle_line - (multiplier * atr)
    #     return upper_line, middle_line, lower_line
    # df['keltner_high'], df['keltner_mid'], df['keltner_low'] = keltner_channel(df['high'], df['low'], df['close'])

    # # Custom indicators
    # # 22. SMA Crossover
    # df['sma_crossover'] = np.where(df['ma_7'] > df['ma_28'], 1, 0)

    # # 23. Price distance from SMA
    # df['price_sma_diff'] = (df['close'] - df['ma_14']) / df['ma_14']

    # # 24. Volatility ratio
    # df['volatility_ratio'] = df['atr'] / df['close']

    # # 25. Price change ratio
    # df['price_change_ratio'] = df['close'] / df['open']
    
    # # 26. Directional Movement Index (DMI)
    # df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    # df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    # df['dx'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=14)

    # # 27. Kaufman's Adaptive Moving Average (KAMA)
    # df['kama'] = talib.KAMA(df['close'], timeperiod=30)

    # # 28. Fractal Adaptive Moving Average (FRAMA)
    # def frama(close, period=16, FC=1, SC=200):
    #     n = (period - 1) // 2
    #     hh = close.rolling(n).max()
    #     ll = close.rolling(n).min()
    #     N1 = (hh - ll) / period
    #     N2 = N1.shift(n)
    #     N3 = (hh.shift(n) - ll.shift(n)) / period
    #     D = (np.log(N1 + N2) - np.log(N3)) / np.log(2)
    #     alpha = np.exp(-4.6 * (D - 1))
    #     alpha = alpha.clip(FC/SC, 1)
    #     frama = close.copy()
    #     for i in range(period, len(close)):
    #         frama.iloc[i] = alpha.iloc[i] * close.iloc[i] + (1 - alpha.iloc[i]) * frama.iloc[i-1]
    #     return frama

    # df['frama'] = frama(df['close'])

    # # 29. Volume Weighted Average Price (VWAP)
    # df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # # 30. Elder's Force Index
    # df['elder_force_index'] = (df['close'] - df['close'].shift(1)) * df['volume']
    # df['elder_force_index_13'] = df['elder_force_index'].ewm(span=13, adjust=False).mean()

    # # 31. Accumulation/Distribution Line (ADL)
    # df['adl'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])

    # # 32. Coppock Curve
    # roc1 = talib.ROC(df['close'], timeperiod=14)
    # roc2 = talib.ROC(df['close'], timeperiod=11)
    # df['coppock'] = (roc1 + roc2).ewm(span=10, adjust=False).mean()

    # # 33. Ease of Movement
    # emv = ((df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2) / (df['volume'] / (df['high'] - df['low']))
    # df['emv'] = emv.rolling(window=14).mean()

    # # 34. Mass Index
    # def mass_index(high, low, period=25, ema_period=9):
    #     amplitude = high - low
    #     ema1 = amplitude.ewm(span=ema_period, adjust=False).mean()
    #     ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
    #     mass = ema1 / ema2
    #     return mass.rolling(window=period).sum()
    
    # df['mass_index'] = mass_index(df['high'], df['low'])
    
    
    # # Price action features
    # df['body'] = df['close'] - df['open']
    # df['wick_upper'] = df['high'] - np.maximum(df['close'], df['open'])
    # df['wick_lower'] = np.minimum(df['close'], df['open']) - df['low']

    # # Candlestick patterns
    # df['doji'] = np.where(np.abs(df['body']) / (df['high'] - df['low']) < 0.1, 1, 0)
    # df['hammer'] = np.where((df['wick_lower'] > 2*np.abs(df['body'])) & (df['wick_upper'] < np.abs(df['body'])), 1, 0)
    # df['shooting_star'] = np.where((df['wick_upper'] > 2*np.abs(df['body'])) & (df['wick_lower'] < np.abs(df['body'])), 1, 0)
    
    # df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
    # df['volume_price_interaction'] = df['volume'] * df['close']
    
    # # Use this function after calculating all other features
    # df = add_lagged_features(df, ['close', 'volume', 'rsi_14', 'macd'])
    
    return df

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the dataframe

    Args:
        df (pd.DataFrame): The input dataframe with the original features

    Returns:
        pd.DataFrame: The dataframe with the temporal features added.
    """
    df['hour'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.hour
    df['day'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.day
    df['month'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.month
    df['year'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.year
    df['day_of_week'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.dayofweek
    df["weekday"] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.weekday
    df['is_weekend'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.day_name().isin(['Saturday', 'Sunday'])
    return df

def add_technical_indicators_volume(df: pd.DataFrame) -> pd.DataFrame:
     # 1. AD
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])

    # 2. ADOSC
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    # 3. OBV
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    return df

def add_technical_indicators_momentum(df: pd.DataFrame) -> pd.DataFrame:
    # 1. ADX
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # 2. ADXR
    df['ADXR'] = talib.ADXR(df['high'], df['low'], df['close'], timeperiod=14)

    # 3. APO
    df['APO'] = talib.APO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    
    # 4. AROON
    aroon_up, aroon_down = talib.AROON(df['high'], df['low'], timeperiod=14)
    df['AROON_Up'] = aroon_up
    df['AROON_Down'] = aroon_down

    # 5. AROONOSC
    df['AROONOSC'] = talib.AROONOSC(df['high'], df['low'], timeperiod=14)
    
    # 6. BOP
    df['BOP'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])

    # 7. CCI
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 8. CMO
    df['CMO'] = talib.CMO(df['close'], timeperiod=14)

    # 9. DX
    df['DX'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=14)

    # 10. MACD
    macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # 11. MACDEXT
    # df['MACDEXT'] = talib.MACDEXT(df['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

    # breakpoint()

    # 12. MACDFIX
    # df['MACDFIX'] = talib.MACDFIX(df['close'], signalperiod=9)

    # 13. MFI
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    
    # 14. MINUS_DI
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

    # 15. MINUS_DM
    df['MINUS_DM'] = talib.MINUS_DM(df['high'], df['low'], timeperiod=14)

    # 16. MOM
    df['MOM'] = talib.MOM(df['close'], timeperiod=14)
    
    # 17. PLUS_DI
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

    # 18. PLUS_DM
    df['PLUS_DM'] = talib.PLUS_DM(df['high'], df['low'], timeperiod=14)

    # 19. PPO
    df['PPO'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    
    # 20. ROC
    df['ROC'] = talib.ROC(df['close'], timeperiod=14)

    # 21. ROCP
    df['ROCP'] = talib.ROCP(df['close'], timeperiod=14)

    # 22. ROCR
    df['ROCR'] = talib.ROCR(df['close'], timeperiod=14)
    
    # 23. ROCR100
    df['ROCR100'] = talib.ROCR100(df['close'], timeperiod=14)

    # 24. RSI
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)

    # 25. STOCH
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['Stoch_K'] = slowk
    df['Stoch_D'] = slowd
    # 26. STOCHF
    df['StochF_K'], df['StochF_D'] = talib.STOCHF(df['high'], df['low'], df['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
    # df['StochF_D'] = talib.STOCHF(df['high'], df['low'], df['close'], fastk_period=5, fastd_period=3, fastd_matype=0)

    # 27. STOCHRSI
    df['StochRSI_K'], df['StochRSI_D'] = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

    # 28. TRIX
    df['TRIX'] = talib.TRIX(df['close'], timeperiod=14)

    # 29. ULTOSC
    df['ULTOSC'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # 30. WILLR
    df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    
    return df

def add_technical_indicators_volatility(df: pd.DataFrame) -> pd.DataFrame:
     # 1. AD
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    
    # 2. ATR
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 3. NATR
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 4. TRANGE
    df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])
    
    return df

def add_technical_indicators_overlap_studies(df: pd.DataFrame) -> pd.DataFrame:
    # 1. BBANDS
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    
    # 2. DEMA
    df['DEMA'] = talib.DEMA(df['close'], timeperiod=30)

    # 3. EMA
    # df['EMA'] = talib.EMA(df['close'], timeperiod=30)
    df['EMA_7'] = talib.EMA(df['close'], timeperiod=7)
    df['EMA_14'] = talib.EMA(df['close'], timeperiod=14)
    df['EMA_28'] = talib.EMA(df['close'], timeperiod=28)

    # 4. HT_TRENDLINE
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close'])

    # 5. KAMA
    df['KAMA'] = talib.KAMA(df['close'], timeperiod=30)

    # 6. MA
    df['MA'] = talib.MA(df['close'], timeperiod=30)

    # 7. MAMA
    mama, fama = talib.MAMA(df['close'])
    df['MAMA'] = mama
    df['FAMA'] = fama

    # 8. MIDPOINT
    df['MIDPOINT'] = talib.MIDPOINT(df['close'], timeperiod=30)

    # 9. MIDPRICE
    df['MIDPRICE'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=30)

    # 10. SAR
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0, maximum=0)

    # 11. SAREXT
    df['SAREXT'] = talib.SAREXT(df['high'], df['low'], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    # 12. SMA
    df['SMA_7'] = talib.SMA(df['close'], timeperiod=7)
    df['SMA_14'] = talib.SMA(df['close'], timeperiod=14)
    df['SMA_28'] = talib.SMA(df['close'], timeperiod=28)

    # 13. T3
    df['T3'] = talib.T3(df['close'], timeperiod=5, vfactor=0)

    # 14. TEMA
    df['TEMA'] = talib.TEMA(df['close'], timeperiod=30)

    # 15. TRIMA
    df['TRIMA'] = talib.TRIMA(df['close'], timeperiod=30)       
    
    # 16. WMA
    df['WMA'] = talib.WMA(df['close'], timeperiod=30)
    
    return df
    
def add_lagged_features(df: pd.DataFrame, columns: list, lag_periods: list = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Add lagged versions of all features in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with features
        lag_periods (list): List of periods to lag

    Returns:
        pd.DataFrame: The dataframe with lagged features added
    """
    for col in columns:
        for lag in lag_periods:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Remove NaN values created by lagging
    # df.dropna(inplace=True)
    
    return df

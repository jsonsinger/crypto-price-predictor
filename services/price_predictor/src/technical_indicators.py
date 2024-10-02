import numpy as np
import pandas as pd
import talib

#class TechnicalIndicators:
    
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

    # 1. Calculate a simple moving average (SMA): Measures the arithmetic mean of a given set of values over a specified period and gives equal weight to each value.
    df['ma_7'] = talib.SMA(df['close'], timeperiod=7)
    df['ma_14'] = talib.SMA(df['close'], timeperiod=14)
    df['ma_28'] = talib.SMA(df['close'], timeperiod=28)

    # 2. Calculate a exponential moving average (EMA): Similar to SMA but places more weight on recent data, making it more responsive to recent changes in price.
    df['ema_7'] = talib.EMA(df['close'], timeperiod=7)
    df['ema_14'] = talib.EMA(df['close'], timeperiod=14)
    df['ema_28'] = talib.EMA(df['close'], timeperiod=28)

    # 3. Relative Strength Index (RSI): Measures the speed and change of price movements, helping to identify overbought or oversold conditions
    df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_28'] = talib.RSI(df['close'], timeperiod=28)

    # 4. Moving Average Convergence Divergence (MACD): Shows the relationship between two moving averages of a price, useful for identifying trend direction and momentum.
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

    # 5. Bollinger Bands: Helps measure market volatility and identify overbought or oversold conditions.
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(df['close'])

    # 6. Stochastic Oscillator: Compares a closing price to its price range over a period of time, useful for identifying potential reversal points.
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])

    # 7. On-Balance Volume (OBV): Relates volume to price change, potentially predicting price movements based on volume trends.
    df['obv'] = talib.OBV(df['close'], df['volume'])

    # 10. Average True Range (ATR): Measures market volatility, which can be useful for setting stop-loss orders or identifying potential breakouts.
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'])

    # 11. Commodity Channel Index (CCI): Helps identify cyclical trends and can signal overbought or oversold conditions.
    df['cci_7'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=7)
    df['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['cci_28'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=28)

    # 12.Money Flow Index (MFI): Combines price and volume data to measure buying and selling pressure.
    df['mfi_7'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=7)
    df['mfi_14'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['mfi_28'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=28)

    # 13. Chaikin Money Flow (CMF): Measures the buying and selling pressure over a specific period, useful for confirming price movements.
    df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    # 14. Rate of Change (ROC): Measures the percentage change in price over a specific period, helping to identify momentum.
    df['roc_7'] = talib.ROC(df['close'], timeperiod=7)
    df['roc_14'] = talib.ROC(df['close'], timeperiod=14)
    df['roc_28'] = talib.ROC(df['close'], timeperiod=28)

    # 15. Williams %R: Similar to the Stochastic Oscillator, it can help identify overbought and oversold levels.
    df['willr_7'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=7)
    df['willr_14'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['willr_28'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=28)

    # 16. Parabolic SAR: Useful for determining potential reversals in price direction.
    df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)

    # 17. Ichimoku Cloud (Conversion Line and Base Line): A comprehensive indicator that provides information about support, resistance, momentum, and trend direction.
    # - ichimoku_conversion: Calculates the average of the 9-period high and 9-period low.
    # - ichimoku_base: Calculates the average of the 26-period high and 26-period low.
    #df['ichimoku_conversion'] = talib.TENKAN_SEN(df['high'], df['low'], timeperiod=9)
    #df['ichimoku_base'] = talib.KIJUN_SEN(df['high'], df['low'], timeperiod=26)
    df['ichimoku_conversion'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['ichimoku_base'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2

    # 18. Average Directional Index (ADX): Measures the strength of a trend
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # 19. Aroon Oscillator: Identifies the start of a new trend and its strength
    df['aroon_osc'] = talib.AROONOSC(df['high'], df['low'], timeperiod=14)

    # 20. Percentage Price Oscillator (PPO): Shows the relationship between two moving averages
    df['ppo'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)

    # 21. Keltner Channel: Similar to Bollinger Bands but uses ATR instead of standard deviation
    def keltner_channel(high, low, close, period=20, atr_period=10, multiplier=2):
        typical_price = (high + low + close) / 3
        middle_line = talib.EMA(typical_price, timeperiod=period)
        atr = talib.ATR(high, low, close, timeperiod=atr_period)
        upper_line = middle_line + (multiplier * atr)
        lower_line = middle_line - (multiplier * atr)
        return upper_line, middle_line, lower_line
    df['keltner_high'], df['keltner_mid'], df['keltner_low'] = keltner_channel(df['high'], df['low'], df['close'])

    # Custom indicators
    # 22. SMA Crossover
    df['sma_crossover'] = np.where(df['ma_7'] > df['ma_28'], 1, 0)

    # 23. Price distance from SMA
    df['price_sma_diff'] = (df['close'] - df['ma_14']) / df['ma_14']

    # 24. Volatility ratio
    df['volatility_ratio'] = df['atr'] / df['close']

    # 25. Price change ratio
    df['price_change_ratio'] = df['close'] / df['open']
    
    # 26. Directional Movement Index (DMI)
    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['dx'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=14)

    # 27. Kaufman's Adaptive Moving Average (KAMA)
    df['kama'] = talib.KAMA(df['close'], timeperiod=30)

    # 28. Fractal Adaptive Moving Average (FRAMA)
    def frama(close, period=16, FC=1, SC=200):
        n = (period - 1) // 2
        hh = close.rolling(n).max()
        ll = close.rolling(n).min()
        N1 = (hh - ll) / period
        N2 = N1.shift(n)
        N3 = (hh.shift(n) - ll.shift(n)) / period
        D = (np.log(N1 + N2) - np.log(N3)) / np.log(2)
        alpha = np.exp(-4.6 * (D - 1))
        alpha = alpha.clip(FC/SC, 1)
        frama = close.copy()
        for i in range(period, len(close)):
            frama.iloc[i] = alpha.iloc[i] * close.iloc[i] + (1 - alpha.iloc[i]) * frama.iloc[i-1]
        return frama

    df['frama'] = frama(df['close'])

    # 29. Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # 30. Elder's Force Index
    df['elder_force_index'] = (df['close'] - df['close'].shift(1)) * df['volume']
    df['elder_force_index_13'] = df['elder_force_index'].ewm(span=13, adjust=False).mean()

    # 31. Accumulation/Distribution Line (ADL)
    df['adl'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])

    # 32. Coppock Curve
    roc1 = talib.ROC(df['close'], timeperiod=14)
    roc2 = talib.ROC(df['close'], timeperiod=11)
    df['coppock'] = (roc1 + roc2).ewm(span=10, adjust=False).mean()

    # 33. Ease of Movement
    emv = ((df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2) / (df['volume'] / (df['high'] - df['low']))
    df['emv'] = emv.rolling(window=14).mean()

    # 34. Mass Index
    def mass_index(high, low, period=25, ema_period=9):
        amplitude = high - low
        ema1 = amplitude.ewm(span=ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
        mass = ema1 / ema2
        return mass.rolling(window=period).sum()
    
    df['mass_index'] = mass_index(df['high'], df['low'])
    
    
    # Price action features
    df['body'] = df['close'] - df['open']
    df['wick_upper'] = df['high'] - np.maximum(df['close'], df['open'])
    df['wick_lower'] = np.minimum(df['close'], df['open']) - df['low']

    # Candlestick patterns
    df['doji'] = np.where(np.abs(df['body']) / (df['high'] - df['low']) < 0.1, 1, 0)
    df['hammer'] = np.where((df['wick_lower'] > 2*np.abs(df['body'])) & (df['wick_upper'] < np.abs(df['body'])), 1, 0)
    df['shooting_star'] = np.where((df['wick_upper'] > 2*np.abs(df['body'])) & (df['wick_lower'] < np.abs(df['body'])), 1, 0)
    
    df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
    df['volume_price_interaction'] = df['volume'] * df['close']
    
    # Use this function after calculating all other features
    df = add_lagged_features(df, ['close', 'volume', 'rsi_14', 'macd'])

    # Remove NaN values
    # df.dropna(inplace=True)

    # # 1. Add Simple Moving Average (SMA) with a
    # # - 7-period window
    # # - 14-period window
    # # - 28-period window
    # df['SMA_7'] = talib.SMA(df['close'], timeperiod=7)
    # df['SMA_14'] = talib.SMA(df['close'], timeperiod=14)
    # df['SMA_28'] = talib.SMA(df['close'], timeperiod=28)
    
    # # 2. Add Exponential Moving Average (EMA) with a
    # # - 7-period window
    # # - 14-period window
    # # - 28-period window
    # df['EMA_7'] = talib.EMA(df['close'], timeperiod=7)
    # df['EMA_14'] = talib.EMA(df['close'], timeperiod=14)
    # df['EMA_28'] = talib.EMA(df['close'], timeperiod=28)

    # # 3. Relative Strength Index (RSI)
    # df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)

    # # 4. Moving Average Convergence Divergence (MACD)
    # macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    # df['MACD'] = macd
    # df['MACD_Signal'] = macd_signal

    # # 5. Bollinger Bands
    # upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    # df['BB_Upper'] = upper
    # df['BB_Middle'] = middle
    # df['BB_Lower'] = lower

    # # 6. Stochastic Oscillator
    # slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # df['Stoch_K'] = slowk
    # df['Stoch_D'] = slowd

    # # 7. On-Balance Volume (OBV)
    # df['OBV'] = talib.OBV(df['close'], df['volume'])

    # # 8. Average True Range (ATR)
    # df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # # 9. Commodity Channel Index (CCI)
    # df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

    # # 10. Chaikin Money Flow (CMF)
    # df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    
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

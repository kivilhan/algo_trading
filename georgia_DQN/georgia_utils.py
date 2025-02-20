import numpy as np
import talib
import pandas as pd

def process_df(csv_path):
    # Load the CSV file
    sym_df = pd.read_csv(csv_path)

    # Ensure the Date column is in datetime format
    if 'Date' in sym_df.columns:
        sym_df['Date'] = pd.to_datetime(sym_df['Date'])

    # Sort by date
    sym_df = sym_df.sort_values(by='Date', ascending=True) if 'Date' in sym_df.columns else sym_df

    ### 1. Candlestick Pattern Recognition ###
    patterns = {
        "Doji": talib.CDLDOJI,
        "Engulfing": talib.CDLENGULFING,
        "Hammer": talib.CDLHAMMER,
        "Morning Star": talib.CDLMORNINGSTAR,
        "Evening Star": talib.CDLEVENINGSTAR,
    }

    for pattern_name, pattern_func in patterns.items():
        sym_df[pattern_name] = pattern_func(sym_df['Open'], sym_df['High'], sym_df['Low'], sym_df['Close']) / 100

    ### 2. Technical Indicators ###
    sym_df['SMA_10'] = talib.SMA(sym_df['Close'], timeperiod=10)
    sym_df['SMA_50'] = talib.SMA(sym_df['Close'], timeperiod=50)
    sym_df['EMA_10'] = talib.EMA(sym_df['Close'], timeperiod=10)
    sym_df['EMA_50'] = talib.EMA(sym_df['Close'], timeperiod=50)
    sym_df['RSI_14'] = talib.RSI(sym_df['Close'], timeperiod=14)
    sym_df['MACD'], sym_df['MACD_Signal'], sym_df['MACD_Hist'] = talib.MACD(sym_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # ### 3. Normalization ###
    # price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist']
    # aapl_df[price_columns] = (aapl_df[price_columns] - aapl_df[price_columns].min()) / (aapl_df[price_columns].max() - aapl_df[price_columns].min())

    # ### 4. Windowed Representation ###
    # window_size = 3  
    # feature_columns = ['Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist']

    # for col in feature_columns:
    #     for i in range(1, window_size + 1):
    #         aapl_df[f"{col}_lag{i}"] = aapl_df[col].shift(i)

    # aapl_df = aapl_df.dropna().reset_index(drop=True)

    # Save the processed data
    sym_df.to_csv("processed_aapl_data.csv", index=False)

    print("Feature engineering completed! Processed data saved as 'processed_aapl_data.csv'.")

    return sym_df
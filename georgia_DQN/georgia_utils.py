import numpy as np
import talib
import pandas as pd
from matplotlib import pyplot as plt

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

def mod_vs_base_plot(model, results):
    close_hist = results['used_data']['close_v']
    mod_out = model(results['used_data']['x_v'])

    close_hist = close_hist.to('cpu').detach().numpy()
    mod_out = mod_out.to('cpu').detach().numpy()

    score_thr = 0.1
    ctr_thr = 3

    stack_list = [mod_out[i:len(mod_out) - ctr_thr + i + 1] for i in range(ctr_thr)]
    mod_hist_stack = np.stack(stack_list)

    thr_exceed_pos = np.greater(mod_hist_stack, score_thr).sum(axis=0)
    thr_exceed_neg = np.less(mod_hist_stack, -score_thr).sum(axis=0)
    thr_exceed_not = np.logical_and(
        np.greater(mod_hist_stack, -score_thr),
        np.less(mod_hist_stack, score_thr)
    ).sum(axis=0)

    longs = np.greater_equal(thr_exceed_pos, ctr_thr)
    shorts = np.greater_equal(thr_exceed_neg, ctr_thr)

    close_trim = close_hist[ctr_thr:]

    base_gains = np.ones((len(close_trim), ))
    mod_gains = np.ones((len(close_trim), ))

    handicap = 0
    pos = 0

    for t in range(1, len(close_trim)):
        if longs[t]:
            if pos != 1:
                mod_gains[t] -= handicap
            pos = 1

        elif shorts[t]:
            if pos != -1:
                mod_gains[t] -= handicap
            pos = -1

        elif thr_exceed_not[t]:
            pos = 0

        if pos == 1:
            mod_gains[t] = mod_gains[t - 1] * (close_trim[t] / close_trim[t - 1])

        elif pos == -1:
            mod_gains[t] = mod_gains[t - 1] * (close_trim[t - 1] / close_trim[t])

        else:
            mod_gains[t] = mod_gains[t - 1]

        base_gains[t] = close_trim[t] / close_trim[0]

    fig = plt.figure()
    plt.figure(figsize=(16, 9))
    plt.title("base gains vs model")
    plt.xlabel("days")
    plt.ylabel("gain")

    plt.plot(base_gains, label="base")
    plt.plot(mod_gains, label="mod")

    plt.grid()
    plt.legend()
    plt.show()

    return {'base': base_gains, 'model': mod_gains}
import pandas as pd
import numpy as np

def process_df(df_path, win_ftr=20, win_past=20, gamma=0.9, offset=50):
    processed_df = pd.read_csv(df_path)[offset:]

    # Columns that will be normalized each window
    window_norm = [
        'Open', 'High', 'Low', 'Close', 'Adj Close','SMA_10',
        'SMA_50', 'EMA_10', 'EMA_50'
    ]

    # Columns that will be normalized throughout whole history
    hist_norm = [
        'Volume', 'Doji', 'Engulfing', 'Hammer', 'Morning Star',
        'Evening Star', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    ]

    close = ['Close']
    window_np = processed_df[window_norm].to_numpy()
    hist_np = processed_df[hist_norm].to_numpy()
    close_np = processed_df[close].to_numpy()

    change_np = close_np[1:] / close_np[:-1]
    stack_list = [change_np[i:len(change_np) - win_ftr + i] for i in range(win_ftr)]
    change_stack = np.concatenate(stack_list, axis=1)
    discounts = np.array([[gamma**i for i in range(win_ftr)]])
    change_discounts = (change_stack - 1) * discounts
    rewards = change_discounts.sum(axis=1)
    rewards = rewards.reshape((len(rewards), 1))[win_past:]
    rewards_norm = rewards / np.abs(rewards).max()

    win_trim = window_np[1:len(window_np) - win_ftr]
    hist_trim = hist_np[1:len(window_np) - win_ftr]
    close_trim = close_np[1:len(window_np) - win_ftr]

    window_norm_np = (win_trim - win_trim.min()) / (win_trim.max() - win_trim.min())

    _min = hist_trim.min(axis=1).reshape(hist_trim.shape[0], 1)
    _max = hist_trim.max(axis=1).reshape(hist_trim.shape[0], 1)
    hist_norm_np = (hist_trim - _min) / (_max - _min)

    df_norm = np.concatenate((window_norm_np, hist_norm_np), axis=1)
    stack_list_2 = [df_norm[i:len(df_norm) - win_past + i] for i in range(win_past)]
    mod_ins = np.stack(stack_list_2, axis=1)

    return {"x": mod_ins, "y": rewards_norm, "close": close_trim}

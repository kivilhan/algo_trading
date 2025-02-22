import pandas as pd
import numpy as np

def process_df(df_path, win_ftr=20, win_past=20, gamma=0.9, offset=50):
    processed_df = pd.read_csv(df_path)[offset:]

    # Columns that will be normalized each window
    window_norm = [
        'Open', 'High', 'Low', 'Close', 'Adj Close','SMA_10',
        'SMA_50', 'EMA_10', 'EMA_50', 
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

def process_df_2(df_path, win_ftr=20, win_past=20, gamma=0.9, offset=50):
    processed_df = pd.read_csv(df_path)[offset:]

    # Columns that will be normalized each window
    window_norm = [
        'Open', 'High', 'Low', 'Close', 'Adj Close','SMA_10',
        'SMA_50', 'EMA_10', 'EMA_50', 
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
    close_trim = close_np[1 + win_past :len(window_np) - win_ftr]

    win_list = [win_trim[i:len(win_trim) - win_past + i] for i in range(win_past)]
    win_stack = np.stack(win_list, axis=1)
    _min = win_stack.min(axis=1).reshape((win_stack.shape[0], 1, win_stack.shape[2]))
    _max = win_stack.max(axis=1).reshape((win_stack.shape[0], 1, win_stack.shape[2]))
    window_norm_np = (win_stack - _min) / (_max - _min)

    _min = hist_trim.min(axis=1).reshape(hist_trim.shape[0], 1)
    _max = hist_trim.max(axis=1).reshape(hist_trim.shape[0], 1)
    hist_norm_np = (hist_trim - _min) / (_max - _min)
    hist_list = [hist_norm_np[i:len(win_trim) - win_past + i] for i in range(win_past)]
    mod_ins = np.stack(hist_list, axis=1)

    return {"x": mod_ins, "y": rewards_norm, "close": close_trim}


def process_df_3(df_path, win_ftr=20, win_past=20, gamma=0.9, offset=50):
    processed_df = pd.read_csv(df_path)[offset:]

    # Set1: Standardise each window with general min-max
    set1 = [
        'Open', 'High', 'Low', 'Close', 'SMA_10',
        'SMA_50', 'EMA_10', 'EMA_50'
        ]

    # Set2: Standardise each window with general min-max
    set2 = ['MACD', 'MACD_Signal', 'MACD_Hist']

    # Set3: Standardise each window individually
    set3 = ['Volume']

    # Set4: Standardise through history individually
    set4 = ['RSI_14'] 

    #####SET1#####
    set1_np = processed_df[set1].to_numpy()
    set1_stack_list = [set1_np[i:len(set1_np) - win_past + i + 1] for i in range(win_past)]
    set1_stack = np.stack(set1_stack_list, axis=1)
    set1_mean = set1_stack.mean(axis=(1, 2)).reshape((len(set1_stack), 1, 1))
    set1_std = set1_stack.std(axis=(1, 2)).reshape((len(set1_stack), 1, 1))
    set1_standard = (set1_stack - set1_mean) / set1_std

    #####SET2#####
    set2_np = processed_df[set2].to_numpy()
    set2_stack_list = [set2_np[i:len(set2_np) - win_past + i + 1] for i in range(win_past)]
    set2_stack = np.stack(set2_stack_list, axis=1)
    set2_mean = set2_stack.mean(axis=(1, 2)).reshape((len(set2_stack), 1, 1))
    set2_std = set2_stack.std(axis=(1, 2)).reshape((len(set2_stack), 1, 1))
    set2_standard = (set2_stack - set2_mean) / set2_std

    #####SET3#####
    set3_np = processed_df[set3].to_numpy()
    set3_stack_list = [set3_np[i:len(set3_np) - win_past + i + 1] for i in range(win_past)]
    set3_stack = np.stack(set3_stack_list, axis=1)
    set3_mean = set3_stack.mean(axis=1).reshape((len(set3_stack), 1, 1))
    set3_std = set3_stack.std(axis=1).reshape((len(set3_stack), 1, 1))
    set3_standard = (set3_stack - set3_mean) / set3_std

    #####SET4#####
    set4_np = processed_df[set4].to_numpy()
    set4_mean = set4_np.mean(axis=0).reshape(1, set4_np.shape[1])
    set4_std = set4_np.std(axis=0).reshape(1, set4_np.shape[1])
    set4_standard = (set4_np - set4_mean) / set4_std
    set4_stack_list = [set4_standard[i:len(set4_standard) - win_past + i + 1] for i in range(win_past)]
    set4_stack = np.stack(set4_stack_list, axis=1)

    #####INPUT TENSOR#####
    model_inputs = np.concatenate((
        set1_standard,
        set2_standard,
        set3_standard,
        set4_stack
    ), axis=2)[:-win_ftr]

    #####LABELS#####
    close_np = processed_df['Close'].to_numpy()
    close_dif = close_np[win_past:len(close_np)] / close_np[win_past - 1:len(close_np) - 1]
    dif_stack_list = [close_dif[i:len(close_dif) - win_ftr + i + 1] for i in range(win_ftr)]
    dif_stack = np.stack(dif_stack_list, axis=1)
    discounts = np.array([[gamma**i for i in range(win_ftr)]])
    dif_discounts = (dif_stack - 1) * discounts
    rewards = dif_discounts.sum(axis=1).reshape((len(dif_discounts), 1))
    rewards_standard = rewards / rewards.std()

    #####REAL PRICES#####
    close_trim = close_np[win_past - 1:len(close_np) - win_ftr]

    return {"x": model_inputs, "y": rewards_standard, "close": close_trim}
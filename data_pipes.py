import numpy as np
import yfinance as yf
import constants

def nova_data(symbols, df):
    close = df["Close"]

    # Check NaN ratios
    for sym in symbols:
        sym_close = df["Close"][sym].to_numpy()
        nan_ratio = np.isnan(sym_close).sum() / len(sym_close)
        if nan_ratio > 0.1:
            print(f"{sym} nan ratio: {nan_ratio:.4f} - dropping")
            close = close.drop(sym, axis=1)

    # Convert to np array
    close_np = close.to_numpy()
    x, y, index = [], [], []
    for i in range(len(df.index)):
        if len(df.index) - i > 6:
            c0 = df.index[i].hour == 9
            c1 = df.index[i+3].hour == 12
            c2 = df.index[i+6].hour == 15
            c3 = np.logical_not(np.any(np.isnan(close_np[i])))
            c4 = np.logical_not(np.any(np.isnan(close_np[i+3])))
            c5 = np.logical_not(np.any(np.isnan(close_np[i+6])))
            if c0 and c1 and c2 and c3 and c4 and c5:
                x.append(close_np[i+3] / close_np[i] - 1)
                y.append(close_np[i+6] / close_np[i+3] - 1)
                index.append(df.index[i])

    x_np = np.stack(x)
    y_np = np.stack(y)

    x_std = (x_np - x_np.mean()) / x_np.std() / 2
    y_std = (y_np - y_np.mean()) / y_np.std() / 2

    x_norm = x_np * 50
    y_norm = y_np * 50

    # return {"x": x_std, "y": y_std}
    return {"x": x_norm, "y": y_norm, "index": index}

def hanzo_df_array(symbols, df):
    close = df["Close"]
    vol = df["Volume"]

    # Check NaN ratios
    for sym in symbols:
        sym_close = df["Close"][sym].to_numpy()
        sym_vol = df["Volume"][sym].to_numpy()
        nan_ratio_close = np.isnan(sym_close).sum() / len(sym_close)
        nan_ratio_vol = np.isnan(sym_vol).sum() / len(sym_vol)
        nan_ratio = max(nan_ratio_close, nan_ratio_vol)
        if nan_ratio > 0.1:
            print(f"{sym} nan ratio: {nan_ratio:.4f} - dropping")
            close = close.drop(sym, axis=1)
            vol = vol.drop(sym, axis=1)

    # Convert to np array
    close_np, vol_np = close.to_numpy(), vol.to_numpy()
    hist, future = 35, 7
    x, y = [], []

    for idx in range(hist+1, len(close_np)-future):
        p0 = close_np[idx - hist - 1:idx - 1]
        p1 = close_np[idx - hist:idx]
        p_idx = p1 / p0 - 1

        v0 = vol_np[idx - hist - 1:idx - 1]
        v1 = vol_np[idx - hist:idx]
        v_idx = v1 / v0 - 1

        x_idx = np.stack((p_idx.T, v_idx.T), axis=2)

        y0 = close_np[idx]
        y1 = close_np[idx + future]
        y_idx = np.array(([y1 / y0 - 1])).T

        if not np.any(np.isnan(x_idx)) and not np.any(np.isnan(y_idx)):
            x.append(x_idx)
            y.append(y_idx)

    x_np = np.stack(x, axis=0)
    y_np = np.stack(y, axis=0)

    return { "x": x_np, "y": y_np }
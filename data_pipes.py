import numpy as np
import yfinance as yf
import constants

def nova_pipe():
    # Get symbols from constants file
    lines = constants.sa_str.splitlines()
    symbols = [line.split("\t")[1] for line in lines][:100]

    # Download data
    df = yf.download(symbols + ["SPY"], period="1y", interval="1h", ignore_tz=True)
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
    x, y = [], []
    for i in range(len(df.index)):
        if len(df.index) - i > 6:
            c0 = df.index[i].hour == 9
            c1 = df.index[i+3].hour == 12
            c2 = df.index[i+6].hour == 15
            if c0 and c1 and c2:
                x.append(close_np[i+3] / close_np[i] - 1)
                y.append(close_np[i+6] / close_np[i+3] - 1)

    x_np = np.stack(x)
    y_np = np.stack(y)
    x_std = (x_np - x_np.mean()) / x_np.std() / 2
    y_std = (y_np - y_np.mean()) / y_np.std() / 2

    return {"x": x_std, "y":}
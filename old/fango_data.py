import yfinance as yf
import numpy as np
import fango_constants as fcs
import pickle
import os
from datetime import datetime

def yf2dct(period="max"):
    play_df = yf.download(fcs.play_list, period=period, interval="1h")
    data_dict = {}
    for key in ["High", "Low", "Close", "Volume"]:
        sym_list = [play_df[key, sym] for sym in fcs.play_list]
        tmp_array = np.array(sym_list, np.float32).T
        for t in range(1, tmp_array.shape[0]):
            for i in range(tmp_array.shape[1]):
                if np.isnan(tmp_array[t, i]):
                    tmp_array[t, i] = tmp_array[t - 1, i]
                    
        data_dict[key] = np.array(tmp_array, np.float32)
        
    return data_dict

def dct2tdata(data_dict, hist=72, future=35):
    x_list = []
    y_list = []
    for t in range(hist, data_dict["High"].shape[0] - future):
        rng_x = range(t - hist, t)
        hi_x = data_dict["High"][rng_x]
        lo_x = data_dict["Low"][rng_x]
        vol = data_dict["Volume"][rng_x]
        norm_hi = (hi_x - lo_x.min()) / (hi_x.max() - lo_x.min())
        norm_lo = (lo_x - lo_x.min()) / (hi_x.max() - lo_x.min())
        norm_vol = (vol - vol.min()) / (vol.max() - vol.min())
        x_list.append(np.stack((norm_hi, norm_lo, norm_vol), axis = 0))
        
        rng_y = range(t, t + future)
        hi_y = data_dict["High"][rng_y]
        lo_y = data_dict["Low"][rng_y]
        close = data_dict["Close"][t]
        y = hi_y.max(axis=0) * lo_y.min(axis=0) / close / close
        y_list.append((y - y.min()) / (y.max() - y.min()) * 0.8 + 0.1)
        
    x = np.array(x_list)
    y = np.array(y_list)
    
    candles = {}
    for key in [ "High", "Low", "Close", "Volume"]:
        candles[key] = data_dict[key][hist:data_dict[key].shape[0] - future]
    
    return {"x": x, "y": y, "candles": candles}

def dct2tdata_dif(data_dict, hist=72, future=35):
    x_list = []
    y_list = []
    dif_dict = data_dict.copy()
    l = dif_dict["High"].shape[0]
    for key in ["High", "Low"]:
        a = dif_dict[key]
        a_dif = (a[1:] - a[:l - 1]) / a[:l - 1]
        dif_dict[key] = (a_dif - a_dif.mean(axis=0)) / a_dif.std(axis=0)
        
    for t in range(hist, data_dict["High"].shape[0] - future):
        rng_x = range(t-hist, t)
        hi_x = dif_dict["High"][rng_x]
        lo_x = dif_dict["Low"][rng_x]
        vol = dif_dict["Volume"][rng_x]
        norm_vol = (vol - vol.min()) / (vol.max() - vol.min())
        x_list.append(np.stack((hi_x, lo_x, norm_vol), axis = 0))
        
        rng_y = range(t, t + future)
        hi_y = data_dict["High"][rng_y]
        lo_y = data_dict["Low"][rng_y]
        close = data_dict["Close"][t]
        y = hi_y.max(axis=0) * lo_y.min(axis=0) / close / close
        y_list.append((y - y.min()) / (y.max() - y.min()) * 0.8 + 0.1)
        
    x = np.array(x_list)
    y = np.array(y_list)
    
    candles = {}
    for key in [ "High", "Low", "Close", "Volume"]:
        candles[key] = data_dict[key][hist:data_dict[key].shape[0] - future]
    
    return {"x": x, "y": y, "candles": candles}

def shuffleData(data):
    x, y = data["x"], data["y"]
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    data["x"], data["y"] = x[idx], y[idx] 
    
    return data

def splitData(data, ratio):
    x, y = data["x"], data["y"]
    split_idx = int(x.shape[0] * ratio)
    candles_train, candles_test = {}, {}
    for key in [ "High", "Low", "Close", "Volume"]:
        candles_train[key] = data["candles"][key][:split_idx]
        candles_test[key] = data["candles"][key][split_idx:]
    out ={
        "x": x[:split_idx],
        "y": y[:split_idx],
        "x_v": x[split_idx:],
        "y_v": y[split_idx:],
        "candles": candles_train,
        "candles_v": candles_test
    }
    
    return out

def save_file(obj, file_name, add_time = True):
    if add_time:
        now = datetime.now()
        stamp = now.strftime("%m%d%y-%H%M%S")
        file_name += "_" + stamp
        
    path = os.path.join("files", file_name + ".pickle")
    with open(path, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
    if add_time:
        return stamp

def load_file(file_name):
    path = os.path.join("files", file_name + ".pickle")
    with open(path, "rb") as file:
        obj = pickle.load(file)
        
    return obj
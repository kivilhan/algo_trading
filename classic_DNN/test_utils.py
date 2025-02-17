import numpy as np
import torch
from matplotlib import pyplot

def plt_single_t_multi_sym(mod, x, y):
    pred = mod(x)


def score_0(model, dict_path, data):
    model.load_state_dict(torch.load(dict_path, weights_only=True))
    x, y = data["x"], data["y"]
    mod_out = model(x)
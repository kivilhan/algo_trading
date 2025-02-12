import pickle
import os

def save_model(model, name):
    folder_path = "machines"
    path = os.path.join(folder_path, name)

    with open(path, "wb") as f:
        pickle.dump(model.state_dict(), f, pickle.HIGHEST_PROTOCOL)

def load_model(base_model, name):
    folder_path = "machines"
    path = os.path.join(folder_path, name)

    with open(path, "rb") as f:
        state_dict = pickle.load(f)

    base_model.load_state_dict(state_dict)

    return base_model
import random
import torch
from torch import nn
import numpy as np

def train_loop(x, y, model, loss_fn, optimizer, batch_size):
    size = x.shape[0]
    num_batches = size // batch_size
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch in range(num_batches):
        # Compute prediction and loss
        batch_start, batch_end = batch * batch_size, (batch + 1) * batch_size
        x_batch, y_batch = x[batch_start:batch_end], y[batch_start:batch_end]
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 25 == 0:
        if batch == num_batches - 1:
            loss, current = loss.item(), batch * batch_size + len(x_batch)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        return loss
    
def test_loop(x, y, model, loss_fn, batch_size):
    size = x.shape[0]
    num_batches = max(1, size // batch_size)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch in range(num_batches):
            batch_start, batch_end = batch * batch_size, (batch + 1) * batch_size
            x_batch, y_batch = x[batch_start:batch_end], y[batch_start:batch_end]
            pred = model(x_batch)
            test_loss += loss_fn(pred, y_batch).item()

    test_loss /= num_batches
    # print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    return test_loss

def random_model_config():
    num_layers = random.randint(1, 10)  # Random number of layers between 1 and 10
    neurons = [random.randint(4, 512) for _ in range(num_layers - 1)] + [1]  # Last neuron always 1
    activations = [random.choice(['relu', 'selu', 'sigmoid', 'none']) for _ in range(num_layers)]
    dropouts = [round(random.uniform(0.0, 0.5), 2) for _ in range(num_layers - 1)] + [0.0]  # Last dropout is 0

    return {
        "neurons": neurons,
        "activations": activations,
        "dropouts": dropouts
    }

def rate_schedule_cosine(rate_min = 1e-5,
                         rate_max = 1e-2,
                         epochs = 100,
                         period = 20,
                         decay_param = 1e-3):
    cos_base = (np.cos(np.arange(epochs) / period * 2 * np.pi) + 1) / 2
    cos_decay = cos_base * np.exp(-np.arange(epochs) * decay_param)
    cos_mapped = cos_decay * (rate_max - rate_min) + rate_min
    
    return cos_mapped
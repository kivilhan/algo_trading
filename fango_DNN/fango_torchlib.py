import torch
import numpy as np

# Cosine gain scheduler with compensated decay (doesn't decay min rate)
def rate_schedule_cosine(rate_min = 1e-5,
                         rate_max = 1e-2,
                         epochs = 100,
                         period = 20,
                         decay_param = 1e-3):
    cos_base = (np.cos(np.arange(epochs) / period * 2 * np.pi) + 1) / 2
    cos_decay = cos_base * np.exp(-np.arange(epochs) * decay_param)
    cos_mapped = cos_decay * (rate_max - rate_min) + rate_min
    
    return cos_mapped

def train_loop(data, model, loss_fn, optimizer, batch_size):
    x, y = torch.from_numpy(data["x"]), torch.from_numpy(data["y"])
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
    
def test_loop(data, model, loss_fn, batch_size):
    x, y = torch.from_numpy(data["x_v"]), torch.from_numpy(data["y_v"])
    size = x.shape[0]
    num_batches = size // batch_size
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

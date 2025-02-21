from torch import nn

class georgia_0(nn.Module):
    def __init__(self, win_past=20, features=19, dropout_rate=0.3):
        super().__init__()
        self.flat0 = nn.Flatten()

        self.hidden0 = nn.Linear(features*win_past, 128)
        self.act0 = nn.ReLU()
        self.dropout0 = nn.Dropout(dropout_rate)
        
        self.hidden1 = nn.Linear(128, 128)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.hidden2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(128, 1)
    def forward(self, x):
        x = self.flat0(x)
        x = self.dropout0(self.act0(self.hidden0(x)))
        x = self.dropout1(self.act1(self.hidden1(x)))
        x = self.dropout2(self.act2(self.hidden2(x)))
        x = self.output(x)

        return x
    
class georgia_1(nn.Module):
    def __init__(self, config, win_past=20, features=19):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Flatten())

        for idx in range(len(config['neurons'])):
            if config['activations'][idx] == 'relu':
                activation = nn.ReLU()
            elif config['activations'][idx] == 'selu':
                activation = nn.SELU()
            elif config['activations'][idx] == 'sigmoid':
                activation = nn.Sigmoid()
            elif config['activations'][idx] == 'none':
                activation = "none"
            else:
                raise ValueError(f"Unrecognized activation function at index {idx}: {activations[idx]}")

            if idx == 0:
                self.layers.append(nn.Linear(features*win_past, config['neurons'][idx]))
            else:
                self.layers.append(nn.Linear(config['neurons'][idx - 1], config['neurons'][idx]))

            if activation != 'none':
                self.layers.append(activation)

            self.layers.append(nn.Dropout(config['dropouts'][idx]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

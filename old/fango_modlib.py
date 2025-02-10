import torch.nn as nn
from collections import OrderedDict

# Test network
class fango_test_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 1, 4, stride=2, padding=2)
        self.flat0 = nn.Flatten()
        self.act0 = nn.ReLU()
        self.hidden1 = nn.Linear(185, 20)
        self.act1 = nn.SELU()
        self.hidden2 = nn.Linear(20, 10)
        self.act2 = nn.SELU()
        self.output = nn.Linear(10, 8)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act0(self.flat0(self.conv0(x)))
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        # x = self.output(x)
        x = self.act_output(self.output(x))
        return x
    
def fango_test_1():
    model = nn.Sequential(OrderedDict([
    ('flat0', nn.Flatten()),
    ('drop0', nn.Dropout(0.35)),
    ('dense0', nn.Linear(105, 1)),
    # ('act2', nn.SELU()),
    # ('drop1', nn.Dropout(0.2)),
    # ('dense1', nn.Linear(10, 8)),
    ('act3', nn.Sigmoid()),
]))
    
    return model
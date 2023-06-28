from torch import nn
import torch

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input = nn.Linear(6, 256)
        self.hidden1 = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 4)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)
        
        return x

# input = torch.tensor([1, 1, 1, 1, 1, 1]).float()
# m = Network()
# print(m)
# print(m.forward(input))
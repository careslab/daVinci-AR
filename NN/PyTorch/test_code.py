import torch
import torch.nn as nn
import torch.optim as optim

# set random seed for reproducibility
torch.manual_seed(1234)

# define dataset size and dimensions
dataset_size = 1000
input_dim = 6
output_dim = 4

# generate random dataset
inputs = torch.randn(dataset_size, input_dim)
outputs = torch.randn(dataset_size, output_dim)

# define neural network architecture
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

# instantiate neural network
model = Network()

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_pred = model(inputs)
    loss = criterion(y_pred, outputs)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print loss after every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")


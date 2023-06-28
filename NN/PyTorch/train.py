import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ARDataset import ARDataset
from model import Network
import numpy as np

#Training parameters
lr = 0.001
batch_size = 128
epochs = 200

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data = ARDataset("/home/abhishek/catkin_ws/tools/ds4_corrected/millimeter_data.csv")
# data = ARDataset("dummy_dataset.csv")

train_size = int(0.85 * len(data))
test_size = len(data) - train_size

training_data, testing_data = torch.utils.data.random_split(data, [train_size, test_size])

train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True)

model = Network()
model.to(device=device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    losses = []

    for idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device = device)
        y = y.to(device = device)

        # print(x, y)

        scores = model(x)
        loss = loss_fn(scores, y)

        losses.append(loss.item())

        
        loss.backward()

        optimizer.step()
    
    print(f"loss at epoch {epoch} is {sum(losses)/len(losses)}")

gt_vals = []
pred_vals = []
for idx, (x, y) in enumerate(training_data):
    x = x.to(device = device)
    y = y.to(device = device)

    scores = model(torch.unsqueeze(x,0))
    pred_vals.append(scores.detach().numpy())
    gt_vals.append(y.numpy())

gt_vals = np.array(gt_vals)
pred_vals = np.array(pred_vals)

# print(gt_vals[1] * [639, 479, 639, 479])
# print(pred_vals[1]* [639, 479, 639, 479])
print(gt_vals[10])
print(pred_vals[10])



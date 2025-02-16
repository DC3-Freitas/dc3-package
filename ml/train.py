"""
Pytorch training script for the MLP model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from ml.model import MLP_Model
from ml_dataset.dataset import CrystalDataset
import numpy as np
from datetime import datetime

np.random.seed(42)
exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# HYPERPARAMETERS

epochs = 100
batch_size = 512
shuffle_dataset = True
train_val_split = 0.8

print("== MLP Training Script ==")
print(f"Parameters: epochs={epochs}, batch_size={batch_size}, shuffle_dataset={shuffle_dataset}, train_val_split={train_val_split}")

# MODEL

model = MLP_Model()

optim = torch.optim.Adam(
    model.parameters(),
    lr=5e-3,
    betas=(0.9, 0.999),
)

# Dataset

dataset = CrystalDataset("ml_dataset/data")
dataset_size = len(dataset)

if shuffle_dataset:
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
split = int(np.floor(train_val_split * dataset_size))
train_sampler = SubsetRandomSampler(indices[:split])
val_sampler = SubsetRandomSampler(indices[split:])

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler
)

val_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=val_sampler
)

# loss = nn.NLLLoss()
loss = nn.CrossEntropyLoss()

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# TRAINING
best_acc = 0
for epoch in range(epochs):
    print(f"== Epoch {epoch} ==")
    model.train()
    tot_loss = 0
    for i, (x, y) in enumerate(train_loader):
        optim.zero_grad()
        y_hat = model(x.to(device).float())
        # print(y_hat, y)
        l = loss(y_hat, y.to(device).long())
        tot_loss += l.item()
        l.backward()
        optim.step()
        if i % 30 == 0: print(f"--> Batch {i}, Loss: {l.item()}")

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for i, (x, y) in enumerate(val_loader):
            y = y.to(device)
            y_hat = model(x.to(device).float())
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f"Epoch {epoch}, Validation Accuracy: {correct / total}")
        if correct / total > best_acc:
            best_acc = correct / total
            torch.save(model.state_dict(), f"ml/models/model_best.pt")

    # write csv
    with open(f"ml/statistics/loss_{exp_name}.csv", "a") as f:
        f.write(f"{epoch},{tot_loss},{correct/total}\n")
    torch.save(model.state_dict(), f"ml/models/model_{epoch}.pt")
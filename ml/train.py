import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from ml.model import MLP_Model
from ml_dataset.dataset import CrystalDataset

np.random.seed(42)

# HYPERPARAMETERS

epochs = 100
batch_size = 128
shuffle_dataset = True
train_val_split = 0.8

# MODEL

model = MLP_Model()

optim = torch.optim.Adam(
    model_0.parameters(),
    lr=5e-3,
    betas=(0.9, 0.999),
)

# Dataset

dataset = CrystalDataset("ml_dataset/data")
dataset_size = len(dataset)

if shuffle_dataset:
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    dataset = Subset(dataset, indices)
split = int(np.floor(train_val_split * dataset_size))
train_sampler = SubsetRandomSampler(indices[:split])
val_sampler = SubsetRandomSampler(indices[split:])

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    sampler=train_sampler
)

val_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=val_sampler
)

loss = nn.NLLLoss()

# DEVICE
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# TRAINING
best_acc = 0
for epoch in range(epochs):
    print(f"== Epoch {epoch} ==")
    model.train()
    for i, (x, y) in enumerate(train_loader):
        optim.zero_grad()
        y_hat = model(x)
        l = loss(y_hat, y)
        l.backward()
        optim.step()
        print(f"--> Batch {i}, Loss: {l.item()}")

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for i, (x, y) in enumerate(val_loader):
            y_hat = model(x)
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f"Epoch {epoch}, Validation Accuracy: {correct / total}")
        if correct / total > best_acc:
            best_acc = correct / total
            torch.save(model.state_dict(), f"ml/models/model_best.pt")
    torch.save(model_0.state_dict(), f"ml/models/model_{epoch}.pt")
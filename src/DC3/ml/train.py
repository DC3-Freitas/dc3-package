import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from ml.model import MLP_Model
from ml_dataset.dataset import CrystalDataset
import numpy as np
from datetime import datetime
import os
from constants import *

np.random.seed(42)


def train(
    exp_name=None,
    model_pretrained=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle_dataset=SHUFFLE_DATASET,
    train_val_split=TRAIN_VAL_SPLIT,
):
    """
    Train the MLP model on the crystal dataset.
    """
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs("ml/models", exist_ok=True)
    os.makedirs("ml/statistics", exist_ok=True)

    print("== Running MLP Train ==")
    print(
        f"Parameters: epochs={epochs}, batch_size={batch_size}, shuffle_dataset={shuffle_dataset}, train_val_split={train_val_split}"
    )

    # DATASET

    dataset = CrystalDataset("ml_dataset/features")
    dataset_size = len(dataset)

    if shuffle_dataset:
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

    split = int(np.floor(train_val_split * dataset_size))
    train_sampler = SubsetRandomSampler(indices[:split])
    val_sampler = SubsetRandomSampler(indices[split:])

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    # MODEL

    model = MLP_Model(dataset.means, dataset.stds)
    if model_pretrained is not None:
        model.load_state_dict(
            torch.load(model_pretrained, map_location="cpu", weights_only=True)
        )
        print(f"Loaded pretrained model from {model_pretrained}")

    optim = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),
    )

    # LOSS

    loss = nn.NLLLoss()

    # DEVICE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # TRAINING

    best_acc = 0

    for epoch in range(epochs):
        print(f"== Epoch {epoch} ==")

        # Training
        model.train()
        tot_loss = 0

        for i, (x, y) in enumerate(train_loader):
            optim.zero_grad()
            y_hat = model(x.to(device).float())
            l = loss(y_hat, y.to(device).long())
            tot_loss += l.item()
            l.backward()
            optim.step()

            if i % 30 == 0:
                print(f"--> Batch {i}, Loss: {l.item()}")

        # Testing
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
                torch.save(model.state_dict(), f"ml/models/model_{exp_name}.pt")

        # Save stats
        with open(f"ml/statistics/loss_{exp_name}.csv", "a") as f:
            f.write(f"{epoch},{tot_loss},{correct/total}\n")

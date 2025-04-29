import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from DC3.ml.model import MLPModel
from DC3.ml_dataset.dataset import CrystalDataset
import numpy as np
from datetime import datetime
import os
from DC3.constants import EPOCHS, BATCH_SIZE, SHUFFLE_DATASET, TRAIN_VAL_SPLIT, SAVED_DIR, SAVED_SYNTH_FEAT_DIR, SAVED_ML_MODELS_DIR, SAVED_ML_STATS_DIR


def train(
    model: MLPModel,
    dataset: CrystalDataset,
    exp_name=None,
    save_model_dir=None,
    save_stats_dir=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle_dataset=SHUFFLE_DATASET,
    train_val_split=TRAIN_VAL_SPLIT,
):
    """
    TODO
    """

    # 1) Setup

    if exp_name is None:
        exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Empty the file
    if save_stats_dir is not None:
        os.makedirs(save_stats_dir, exist_ok=True)
        
        with open(os.path.join(save_stats_dir, f"{exp_name}.csv"), "w") as f:
            pass

    print("\n== Preparing Trainer ==")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}, shuffle_dataset={shuffle_dataset}, train_val_split={train_val_split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))
    criterion = nn.NLLLoss()

    # 2) Dataset

    indices = np.arange(len(dataset))
    if shuffle_dataset:
        np.random.shuffle(indices)

    split_idx = int(np.floor(train_val_split * len(dataset)))
    train_sampler = SubsetRandomSampler(indices[:split_idx])
    val_sampler = SubsetRandomSampler(indices[split_idx:])

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    best_val_acc = 0.0

    # 3) Train loop

    for epoch in range(1, epochs + 1):
        print(f"\n== Epoch {epoch}/{epochs} ==")

        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device).float(), y.to(device).long()

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 30 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.6f}")

        # 4) Validation

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device).float(), y.to(device).long()
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        # 5) Save

        if save_model_dir is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_model_dir, f"{exp_name}.pth"))

        if save_stats_dir is not None:
            os.makedirs(save_stats_dir, exist_ok=True)
            with open(os.path.join(save_stats_dir, f"{exp_name}.csv"), "a") as f:
                f.write(f"{epoch},{running_loss:.6f},{val_acc:.6f}\n")

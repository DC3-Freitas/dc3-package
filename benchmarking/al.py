from ml.model import MLP_Model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

SIM_TEMPERATURE_FRACTIONS = np.round(np.arange(0.04, 1.60 + 0.04, 0.04), 6)

model = MLP_Model()
model.load_state_dict(torch.load("ml/models/model_99.pt", weights_only=True))

# Eval mode
model.eval()

acc = np.loadtxt("acc.txt")
plt.plot(SIM_TEMPERATURE_FRACTIONS, acc)
plt.show()
exit()

acc = np.zeros_like(SIM_TEMPERATURE_FRACTIONS)

for exp_name in os.listdir("md/features"):
    for file_name in os.listdir(f"md/features/{exp_name}"):
        # Get file info 
        file_path = f"md/features/{exp_name}/{file_name}"
        sim_temp_id = file_name[8:-3]

        # Get data
        data = torch.from_numpy(np.loadtxt(file_path)).float()

        means = torch.from_numpy(np.loadtxt("ml/models/means.txt")).float()
        stds = torch.from_numpy(np.loadtxt("ml/models/stds.txt")).float()
        data = (data - means) / stds

        # Prediction
        with torch.no_grad():
            preds = model(data).argmax(dim=1)
            print(file_path + ":", [(preds == i).sum().item() for i in range(6)])

        correct = (preds == 2).sum().item()
        
        # Log info
        idx = np.where(SIM_TEMPERATURE_FRACTIONS == float(sim_temp_id))[0][0]
        acc[idx] = correct / len(data)

np.savetxt("acc.txt", acc)

plt.plot(SIM_TEMPERATURE_FRACTIONS, acc)
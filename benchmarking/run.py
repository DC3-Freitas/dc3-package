"""
from ml.model import MLP_Model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

SIM_TEMPERATURE_FRACTIONS = np.round(np.arange(0.04, 1.60 + 0.04, 0.04), 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP_Model()
model.load_state_dict(torch.load("ml/models/model_2025-03-23_14-15-09.pt"))
model.to(device)

# Eval mode
model.eval()
#acc = np.loadtxt("acc.txt")
#plt.plot(SIM_TEMPERATURE_FRACTIONS, acc)
#plt.show()
#exit()

res = {}

correct_map = {
    "al_fcc": 2,
    "fe_bcc": 0,
    "mg_hcp": 3,
    "si_cd" : 1,
}

means = torch.from_numpy(np.loadtxt("ml/models/means.txt")).float()
stds = torch.from_numpy(np.loadtxt("ml/models/stds.txt")).float()

for exp_name in os.listdir("md/features"):
    acc = [] #np.zeros_like(SIM_TEMPERATURE_FRACTIONS)
    for file_name in os.listdir(f"md/features/{exp_name}"):
        # Get file info 
        file_path = f"md/features/{exp_name}/{file_name}"
        sim_temp_id = file_name[8:-3]

        # Get data
        data = torch.from_numpy(np.loadtxt(file_path)).float()
        # means = np.mean(data, axis=0)
        # stds = np.std(data, axis=0)
        # data = (data - means) / stds
        # data = torch.from_numpy(data).float()

        # Prediction
        with torch.no_grad():
            preds = model(data.to(device)).argmax(dim=1).cpu()
            print(file_path + ":", [(preds == i).sum().item() for i in range(6)])

        correct = (preds == correct_map[exp_name]).sum().item()
        
        # Log info
        idx = np.where(SIM_TEMPERATURE_FRACTIONS == float(sim_temp_id))[0][0]
        #acc[idx] = correct / len(data)
        acc.append(correct / len(data))
        print(f"Accuracy for {exp_name} at {sim_temp_id}: {correct / len(data)}")
    res[exp_name] = acc

# np.savetxt("acc.txt", acc)
np.save("res.npy", res)

# set up subplots
fig, ax = plt.subplots((len(res) + 1) // 2, 2, figsize=(10, 10))
ctr = 0
for exp_name in res:
    acc = res[exp_name]
    ax[ctr // 2, ctr % 2].plot(SIM_TEMPERATURE_FRACTIONS, acc)
    ax[ctr // 2, ctr % 2].set_title(exp_name)
    ctr += 1

#plt.plot(SIM_TEMPERATURE_FRACTIONS, acc)
plt.show()
"""
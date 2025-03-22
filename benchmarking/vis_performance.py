from ml.model import MLP_Model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

SIM_TEMPERATURE_FRACTIONS = np.round(np.arange(0.04, 1.60 + 0.04, 0.04), 6)

res = np.load("res.npy", allow_pickle=True).item()

# set up subplots
fig, ax = plt.subplots(1, len(res), figsize=(10, 10))
for i, exp_name in enumerate(res):
    acc = res[exp_name]
    ax[i].plot(SIM_TEMPERATURE_FRACTIONS, acc)
    ax[i].set_title(exp_name)

plt.show()
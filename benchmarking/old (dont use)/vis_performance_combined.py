from ml.model import MLP_Model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

SIM_TEMPERATURE_FRACTIONS = np.round(np.arange(0.04, 1.60 + 0.04, 0.04), 6)

res = np.load("benchmarking/old/res.npy", allow_pickle=True).item()
plt.style.use("bmh")

for exp_name in res:
    acc = res[exp_name]
    plt.plot(SIM_TEMPERATURE_FRACTIONS, acc, label=exp_name)

plt.title("Model Performance on MD Data")
plt.xlabel("Simulation Temperature Fraction (T/Tm)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
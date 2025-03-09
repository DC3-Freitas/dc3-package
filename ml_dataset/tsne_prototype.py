"""
Loads .gz files generated in /data for fcc and bcc, then
runs t-SNE visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load(prefix):
    """
    Load data from .gz files in /data

    Args:
        prefix: path to .gz file
    """
    data = []
    for file in os.listdir("ml_dataset/data/"):
        if file.endswith(".gz") and file.startswith(prefix):
            data.append(np.loadtxt(os.path.join("ml_dataset/data", file)))
    data = np.vstack(data)
    return data

if __name__ == "__main__":
    # Load data
    fcc = load("fcc")
    bcc = load("bcc")
    fcc = fcc[~np.isnan(fcc).any(axis=1)][:100]
    bcc = bcc[~np.isnan(bcc).any(axis=1)][:100]
    print("data loaded!")

    # Generate full data and labels
    data = np.vstack((fcc, bcc))
    print("dataset size:", data.shape)
    labels = np.zeros(len(data))
    print("dataset length:", len(data))
    print("fcc:", len(fcc))
    print("bcc:", len(bcc))
    labels[len(fcc):] = 1

    # Run PCA for dimensionality reduction
    print("feature vector length:", len(data[0]))
    #pca = PCA(n_components=50)
    #data = pca.fit_transform(data)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    data_2d = tsne.fit_transform(data)
    plt.style.use("bmh")
    scatter = plt.scatter(data_2d[:,0], data_2d[:,1], c=labels)
    scatter_handles, scatter_labels = scatter.legend_elements()
    plt.legend(handles=scatter_handles, labels=["fcc", "bcc"])
    
    # Save
    plt.savefig("ml_dataset/tsne.png")

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ml_dataset.dataset import CrystalDataset

dataset = CrystalDataset("ml_dataset/data")
data, labels, label_map = dataset.data, dataset.labels, dataset.label_map

# shuffle
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices][:1000]
labels = labels[indices][:1000]

PCA_COMPONENTS = 129

def tsne(data, labels, label_map):
    # Run PCA for dimensionality reduction
    #pca = PCA(n_components=PCA_COMPONENTS)
    #data = pca.fit_transform(data)
    #print("PCA complete!")

    # Run t-SNE
    tsne = TSNE(n_components=1, random_state=0)
    data_2d = tsne.fit_transform(data)
    print("t-SNE complete!")

    # Plot
    plt.style.use("bmh")
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 0], c=labels)
    scatter_handles, scatter_labels = scatter.legend_elements()
    plt.legend(handles=scatter_handles, labels=label_map.keys())
    plt.show()
    plt.savefig("tsne.png")

tsne(data, labels, label_map)
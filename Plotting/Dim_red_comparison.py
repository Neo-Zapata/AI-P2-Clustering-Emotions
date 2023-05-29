import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import (PCA, TruncatedSVD)
import os
import warnings
warnings.filterwarnings('ignore')
import random
random.seed(42)
np.random.seed(42)

cwd         = os.getcwd()

pca         = PCA(n_components = 2, random_state = 0)
SVD         = TruncatedSVD(n_components = 2, algorithm = 'randomized', random_state = 0, n_iter = 5)

dim_reduction_methods = {
    'PCA': pca,
    'SVD': SVD
}

def dim_reduction_methods_comparison(data, labels, centroids, k, path):
    
    _, ax = plt.subplots(nrows = 1, ncols = 2)

    for j,(name, model) in enumerate(dim_reduction_methods.items()):
        data_m = data
        if path != 'wavelet_VC_2.csv':
            data_m = model.fit_transform(data)
        ax[j].scatter(data_m[:, 0], data_m[:, 1], c=labels, s=20, cmap='Set1', label=name)
        ax[j].set_title(name)

    plt.tight_layout()
    save_dir = cwd + f"/images/clusters/{path}_"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"Kmeans_{k}_clusters.png"))
    plt.close()


def dim_reduction_methods_comparison_dbscan(data, labels, k, path):
    
    _, ax = plt.subplots(nrows = 1, ncols = 2)

    for j,(name, model) in enumerate(dim_reduction_methods.items()):
        data_m = data
        if path != 'wavelet_VC_2.csv':
            data_m = model.fit_transform(data)
        ax[j].scatter(data_m[:, 0], data_m[:, 1], c=labels, s=20, cmap='Set1', label=name)
        ax[j].set_title(name)

    plt.tight_layout()
    save_dir = cwd + f"/images/dbscan/{path}_"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"dbscan_cluster_{k}.png"))
    plt.close()


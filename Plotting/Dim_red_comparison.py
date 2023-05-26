import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import (PCA, IncrementalPCA, TruncatedSVD)
import os
import warnings
warnings.filterwarnings('ignore')

cwd                     = os.getcwd()

pca         = PCA(n_components = 2, random_state = 0)
inc_pca     = IncrementalPCA(n_components = 2)
SVD         = TruncatedSVD(n_components = 2, algorithm = 'randomized', random_state = 0, n_iter = 5)

dim_reduction_methods = {
    'PCA': pca,
    'INC PCA': inc_pca,
    'SVD': SVD
}

def dim_reduction_methods_comparison(data, labels, centroids, k, path):
    
    if path != '2_VC.csv':
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 4))
        for j,(name, model) in enumerate(dim_reduction_methods.items()):
            data_m = model.fit_transform(data)
            ax[j].scatter(data_m[:, 0], data_m[:, 1], c=labels, s=20, cmap='Set1', label=name)
            ax[j].set_title(name)
        # plt.tight_layout()
        # # plt.show()
        # save_dir = cwd + f"/images/{path}_"
        # os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_dir, f"Kmeans_{k}_clusters.png"))
        
    else:
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 4))
        for j,(name, model) in enumerate(dim_reduction_methods.items()):
            ax[j].scatter(data[:, 0], data[:, 1], c=labels, s=20, cmap='Set1', label=name)
            ax[j].set_title(name)

    plt.tight_layout()
    # plt.show()
    save_dir = cwd + f"/images/{path}_"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"Kmeans_{k}_clusters.png"))



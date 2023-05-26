import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from scipy.signal import argrelextrema
from sklearn.decomposition import (PCA, IncrementalPCA, TruncatedSVD, NMF)
import os
np.random.seed(42)

pca         = PCA(n_components = 2, random_state = 0)
inc_pca     = IncrementalPCA(n_components = 2)
SVD         = TruncatedSVD(n_components = 2, algorithm = 'randomized', random_state = 0, n_iter = 5)
nmf         = NMF(n_components=2)

dim_reduction_methods = {
    'PCA': pca,
    'INC PCA': inc_pca,
    'SVD': SVD,
    'NMF': nmf
}

cwd                     = os.getcwd()
charact_vec_path        = cwd + "/data/characteristic_vectors"
data_path               = os.listdir(charact_vec_path)

for path in data_path:
    print(f"------------------------------- {path} -------------------------------")
    # Normalizar la data
    data        = np.loadtxt(charact_vec_path + "/" + path, delimiter=',')
    scaler      = MinMaxScaler()
    data        = scaler.fit_transform(data)

    # Get the number oof clusters
    wcss    = []
    sil     = []
    dbs     = []
    rango   = list(range(2,11))
    ks      = []

    for i in rango:
        kmeans = KMeans(n_clusters=i, max_iter=300)
        kmeans.fit(data)
        labels = kmeans.labels_

        wcss.append(kmeans.inertia_)
        sil.append(silhouette_score(data, labels))
        dbs.append(davies_bouldin_score(data, labels))
    
    max = argrelextrema(np.array(sil), np.greater)  # get local max 
    min = argrelextrema(np.array(dbs), np.less)     # get local min
    max = max[0]
    min = min[0]

    ks.extend(max)
    ks.extend(min)
    ks = list(set(ks))
    if 1 in ks:
        ks.remove(1)

    # _, ax = plt.subplots(1, 3)
    # ax[0].plot(range(2, 11), wcss)
    # ax[0].set_title("Jambú elbow")
    # ax[1].plot(range(2, 11), sil)
    # ax[1].set_title('Silhouette Coefficient')
    # ax[2].plot(range(2, 11), dbs)
    # ax[2].set_title("Davies Boulding Score")
    # plt.tight_layout()
    # plt.show()

    print(f"We are gonna try for {ks} clusters")

    for cluster in ks:
        print(f"------------------------------- # Clusters: {cluster} -------------------------------")
        kmeans = KMeans(n_clusters=cluster, max_iter=300)
        kmeans.fit(data)
        labels = kmeans.labels_

        if path != '2_VC.csv':
            fig, ax = plt.subplots(nrows = 1, ncols = 4)
            for j,(name, model) in enumerate(dim_reduction_methods.items()):
                data_m = model.fit_transform(data)
                ax[j].scatter(data_m[:, 0], data_m[:, 1], c=labels, s=20, cmap='Set1', label=name)
                ax[j].set_title(name)
            # plt.show()
            save_dir = cwd + f"/images/{path}_"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"Kmeans_library_{cluster}_clusters.png"))
        
        else:
            fig, ax = plt.subplots(nrows = 1, ncols = 4)
            for j,(name, model) in enumerate(dim_reduction_methods.items()):
                ax[j].scatter(data[:, 0], data[:, 1], c=labels, s=20, cmap='Set1', label=name)
                ax[j].set_title(name)
            # plt.show()
            save_dir = cwd + f"/images/{path}_"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"Kmeans_library_{cluster}_clusters.png"))

        # calculate the quality using ...
        distances = pairwise_distances(data)
        silhouette_avg = silhouette_score(distances, labels)
        print("Índice de validez de silueta:", silhouette_avg)

        # Create an empty similarity matrix
        n = data.shape[1]
        similarity_matrix = np.zeros((n, n))

        # Iterate over data points and fill the similarity matrix
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = 1 / (1 + distances[i, j])

        # plot as a hotmap
        plt.cla()
        plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        
        # Add axis labels
        plt.xlabel('Data Points')
        plt.ylabel('Data Points')

        # Add a colorbar for reference
        plt.colorbar()

        # Show the plot
        # plt.show()

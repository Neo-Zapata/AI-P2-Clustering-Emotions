from image_processing.image_processing import process_dataset
from kmeans.kmeans import Kmeans
from DBSCAN.dbscan import DBSCAN
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from scipy.signal import argrelextrema
warnings.filterwarnings('ignore')

np.random.seed(42)

cwd                     = os.getcwd()
charact_vec_path        = cwd + "/data/characteristic_vectors"

initialization_method   = ['random', 'kmeans++']
distance_metric         = ['euclidean', 'manhattan', 'cosine']
convergence_criterion   = [('umbral', 0.01), ('iteration', 1000)]


if len(os.listdir(charact_vec_path)) == 0:
    process_dataset(5)
 

def get_n_cluster_k_means(data, name, show):
    k_to_try = []
    wcss    = []
    sil     = []
    dbs     = []
    rango   = list(range(2, 10+1))

    # Visualize data
    # if data.shape[1] > 2:
    #     pca_data        = PCA(n_components = 2, random_state = 0)
    #     data            = pca_data.fit_transform(data)

    for k in rango:
        # print(f"------------------------------- {k} clusters -------------------------------")
        kmean_ = Kmeans(k, data, initialization_method[0], distance_metric[0], convergence_criterion[0], False, '')
        kmean_.fit()
        labels = kmean_.labels
        labels = np.array(labels)
        # Jambú Elbow [2-10]        
        wcss.append(kmean_.get_wcss())
        # Silhouette Coefficient
        sil.append(silhouette_score(data, labels))
        # Davies Boulding Score
        dbs.append(davies_bouldin_score(data, labels))

    max = argrelextrema(np.array(sil), np.greater)  # get local max 
    min = argrelextrema(np.array(dbs), np.less)     # get local min
    for index in max[0]:
        # if rango[index] <= max_n_clusters:
        k_to_try.append(rango[index])
    for index in min[0]:
        # if rango[index] <= max_n_clusters:
        k_to_try.append(rango[index])
    # BC we know there should be 7 clusters, we add it if not already in
    # k_to_try.append(7)
    k_to_try = list(set(k_to_try))

    if show:
        _, ax = plt.subplots(2, 2)
        ax[0,0].scatter(data[:, 0], data[:, 1], s=20, cmap='Set1')
        ax[0,0].set_title('Visualize Data')
        ax[0,1].plot(rango, wcss)
        ax[0,1].set_title('Jambú Elbow')
        ax[1,0].plot(rango, sil)
        ax[1,0].set_title('Silhouette Coefficient')
        ax[1,1].plot(rango, dbs)
        ax[1,1].set_title('Davies Boulding Score')
        plt.tight_layout()
        plt.show()
    
    return k_to_try


def test_kmeans(data_path, show_n_cluster_calculation, show_clustring_for_2VC):
    # max_n_clusters = 7
    for path in data_path: # traverse each .csv
        print(f"------------------------------- {path} -------------------------------")
        # Normalizar la data
        data        = np.loadtxt(charact_vec_path + "/" + path, delimiter=',')
        scaler      = MinMaxScaler()
        data        = scaler.fit_transform(data) 

        # Determinar la cantidad de clusters a usar usando diferentes metodos
        n_cluster   = get_n_cluster_k_means(data, path, show=show_n_cluster_calculation)
        print(f"We are gonna try for {n_cluster} clusters")
        for i in n_cluster:
            print(f"------------------------------- # Clusters: {i} -------------------------------")
            kmean_vc = Kmeans(i, data, initialization_method[0], distance_metric[0], convergence_criterion[0], show_clustring_for_2VC, path)
            kmean_vc.fit()
            calculate_quality(kmean_vc)

def calculate_quality(kmean_vc):
    similarity_matrix, distances, silhouette_avg = kmean_vc.silhouette_score()
    print("Índice de validez de silueta:", silhouette_avg)


    # Iterate over data points and fill the similarity matrix
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
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
    plt.show()

def test_dbscan(data_path):
    rad     = 200
    minPts  = 4
    for path in data_path: # traverse each .csv
    # print(f"------------------------------- {path} -------------------------------")
        data = np.loadtxt(charact_vec_path + "/" + path, delimiter=',')
        dbscan_vc = DBSCAN(data, rad, minPts, distance_metric[0], False)
        dbscan_vc.fit()
        if path == '2_VC.csv':
            dbscan_vc.plot_()





data_path   = os.listdir(charact_vec_path)
show_n_cluster_calculation  = False
show_clustring_for_2VC      = False
test_kmeans(data_path, show_n_cluster_calculation, show_clustring_for_2VC)
# test_dbscan(matrixes_path)
    


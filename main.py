from image_processing.image_processing import process_dataset, get_real_labels
from kmeans.kmeans import Kmeans
from DBSCAN.dbscan import DBSCAN
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, confusion_matrix, completeness_score, jaccard_score
from sklearn.metrics.cluster import rand_score
import random
random.seed(42)


np.random.seed(42)

cwd                     = os.getcwd()
charact_vec_path        = cwd + "/data/characteristic_vectors"
initialization_method   = ['random', 'kmeans++']
distance_metric         = ['euclidean', 'manhattan']
convergence_criterion   = [('umbral', 0.01)]


if len(os.listdir(charact_vec_path)) == 0:
    process_dataset(3)
 
real_labels = get_real_labels()

def test_kmeans(files, save_plots):
    # traverse each .csv
    for file in files: 
        print(f"\n------------------------------- {file} -------------------------------")
        data_path   = charact_vec_path + "/" + file

        # get data
        data        = np.loadtxt(data_path, delimiter=',')

        # Determinar la cantidad de clusters a usar usando diferentes metodos
        n_cluster   = get_n_cluster_k_means(data, file)
        print(f"Jambú elbow, Silhouette coefficient and Davies Boulding Score suggest on using  {n_cluster} clusters")

        for i in n_cluster:
            print(f"------------------------------- # Clusters: {i} -------------------------------")
            distances = pairwise_distances(data, metric='manhattan')
            kmean_vc = Kmeans(i, data, initialization_method[1], distance_metric[1], convergence_criterion[0], file, save_plots)
            kmean_vc.fit()
            metrics_evaluation(distances, kmean_vc.labels, file, i, data)


def test_gmm(files, save_plots):
    pass


def test_dbscan(files, save_plots):
# traverse each .csv
    rad = [50, 180, 200, 250]
    minPts = 4
    i = 1
    for file in files: 
        print(f"\n------------------------------- {file} -------------------------------")
        data_path   = charact_vec_path + "/" + file

        # get data
        data        = np.loadtxt(data_path, delimiter=',')
        dbscan = DBSCAN(data, rad[3], minPts, distance_metric[1], save_plots)
        dbscan.fit()
        labels = list(set(dbscan.labels))
        dbscan.plot_(i, file)
        if(len(labels) > 1):
            print(f"Se formaron {dbscan.n_cluster} clusters.")
            metrics_evaluation_dbscan(dbscan.labels, data)
        else:
            print(f"Se formaron {dbscan.n_cluster} clusters. Por lo que, no fue posible realizar el calculo de las metricas.")
        i += 1



def metrics_evaluation(distances, labels, file, k, data):
    plt.clf()
    plt.close()

    # Evaluate clustering using different metrics
    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    rand = rand_score(real_labels, labels)
    jaccard = jaccard_score(real_labels, labels, average='weighted')
    confusion_mat = confusion_matrix(real_labels, labels)
    purity = np.sum(np.amax(confusion_mat, axis=0)) / np.sum(confusion_mat)
    completeness = completeness_score(real_labels, labels)

    silhouette_metric = "BAD" if silhouette < 0.9 else "GOOD"
    rand_metric = "BAD" if rand < 0.9 else "GOOD"
    jaccard_metric = "BAD" if jaccard < 0.9 else "GOOD"
    purity_metric = "BAD" if purity < 0.9 else "GOOD"
    completeness_metric = "BAD" if completeness < 0.9 else "GOOD"

    # Print the evaluation metrics
    print(f"Silhouette Coefficient: [-1 to 1]                            {round(silhouette, 3)}     [{silhouette_metric}]")
    print(f"Calinski-Harabasz Index: [the higher the better]             {round(calinski_harabasz, 3)}")
    print(f"Davies-Bouldin Index: [0 to INF] - lower values are better   {round(davies_bouldin, 3)}")
    print(f"Rand Index: [-INF to 1]                                      {round(rand, 3)}     [{rand_metric}]")
    print(f"Jaccard Coefficient: [0 to 1]                                {round(jaccard, 3)}     [{jaccard_metric}]")
    print(f"Purity: [0 to 1]                                             {round(purity, 3)}     [{purity_metric}]")
    print(f"Completitud: [0 to 1]                                        {round(completeness, 3)}     [{completeness_metric}]")

    # ----------------------------------------------------------------------

    # Display the similarity matrix
    plt.imshow(distances, cmap='hot', interpolation='nearest')

    # Add colorbar for reference
    plt.colorbar()

    # Add labels and title
    plt.xlabel('Samples')
    plt.ylabel('Samples')
    plt.title(f'Similarity Matrix\nSilhouette Score: {silhouette:.2f}')

    plt.tight_layout()
    save_dir = cwd + f"/images/silScore_&_simMatrix/{file}_"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + f'/kmeans_{k}_cluster.png')
    plt.close()

def metrics_evaluation_dbscan(labels, data):
    plt.clf()
    plt.close()

    # Evaluate clustering using different metrics
    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    rand = rand_score(real_labels, labels)
    jaccard = jaccard_score(real_labels, labels, average='weighted')
    confusion_mat = confusion_matrix(real_labels, labels)
    purity = np.sum(np.amax(confusion_mat, axis=0)) / np.sum(confusion_mat)
    completeness = completeness_score(real_labels, labels)

    silhouette_metric = "BAD" if silhouette < 0.9 else "GOOD"
    rand_metric = "BAD" if rand < 0.9 else "GOOD"
    jaccard_metric = "BAD" if jaccard < 0.9 else "GOOD"
    purity_metric = "BAD" if purity < 0.9 else "GOOD"
    completeness_metric = "BAD" if completeness < 0.9 else "GOOD"

    # Print the evaluation metrics
    print(f"Silhouette Coefficient: [-1 to 1]                            {round(silhouette, 3)}     [{silhouette_metric}]")
    print(f"Calinski-Harabasz Index: [the higher the better]             {round(calinski_harabasz, 3)}")
    print(f"Davies-Bouldin Index: [0 to INF] - lower values are better   {round(davies_bouldin, 3)}")
    print(f"Rand Index: [-INF to 1]                                      {round(rand, 3)}     [{rand_metric}]")
    print(f"Jaccard Coefficient: [0 to 1]                                {round(jaccard, 3)}     [{jaccard_metric}]")
    print(f"Purity: [0 to 1]                                             {round(purity, 3)}     [{purity_metric}]")
    print(f"Completitud: [0 to 1]                                        {round(completeness, 3)}     [{completeness_metric}]")

    # ----------------------------------------------------------------------



def get_n_cluster_k_means(data, file):
    k_to_try = []
    wcss     = []
    sil      = []
    dbs      = []
    rango    = list(range(2, 10+1))

    for k in rango:
        kmean_ = Kmeans(k, data, initialization_method[1], distance_metric[1], convergence_criterion[0], '', False)
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
    min = min[0]
    max = max[0]

    k_to_try.extend(max)
    k_to_try.extend(min)
    k_to_try = list(set(k_to_try))
    if 1 in k_to_try:
        k_to_try.remove(1)


    _, ax = plt.subplots(2, 2)
    ax[0,0].scatter(data[:, 0], data[:, 1])
    ax[0,0].set_title('Visualize Data')
    ax[0,1].plot(rango, wcss)
    ax[0,1].set_title('Jambú Elbow')
    ax[1,0].plot(rango, sil)
    ax[1,0].set_title('Silhouette Coefficient')
    ax[1,1].plot(rango, dbs)
    ax[1,1].set_title('Davies Boulding Score')
    plt.tight_layout()
    save_dir = cwd + f"/images/n_cluster_calc"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + f'/{file}_kmean_cluster_calc.png')
    plt.close()
    return k_to_try






csvs   = os.listdir(charact_vec_path)
savel_plots                 = True

print("\n------------------------- TESTING K-MEANS ---------------------------\n")
test_kmeans(csvs, savel_plots)
print("\n------------------------- TESTING DBSCAN ---------------------------\n")
test_dbscan(csvs, savel_plots)
print("\n------------------------- TESTING GMM ---------------------------\n")
test_gmm(csvs, savel_plots)
print("\n Todos los plots generados pueden ubicarse en la carpeta 'images' \n")
    


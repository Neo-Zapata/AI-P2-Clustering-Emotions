import numpy as np
from Plotting.Dim_red_comparison import dim_reduction_methods_comparison
import random
random.seed(42)
np.random.seed(42)



emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

class Kmeans:
    def __init__(self, n_clusters, data_points, initialization_method, distance_metric, convergence_criterion, path, show):
        self.k                  = n_clusters
        self.data               = data_points
        self.init_method        = initialization_method
        self.distance_metric    = distance_metric
        self.convergence_crit   = convergence_criterion
        self.labels             = []
        self.clusters           = {}
        self.centroids          = []
        self.correspondence     = {}
        self.path               = path
        self.show               = show

    def get_wcss(self): # get inerce intra-class
        wcss = []
        for k_index, centroid in enumerate(self.centroids):
            distances = []
            for vc in self.clusters[k_index]:
                distances.append(self.distance(centroid, vc))
            distances = np.array(distances)
            wcss.append(np.mean(distances, axis = 0))
        wcss = np.array(wcss)
        mean = np.mean(wcss, axis = 0)
        return mean

    def fit(self): # fit data to the model
        self.centroids          = self.Init_Centroide()
        self.clusters           = self.get_clusters()
        new_centroides          = self.new_centroide()
        if self.convergence_crit[0] == 'umbral':
            while(self.centroids_distance(new_centroides) > self.convergence_crit[1]):
                self.centroids  = new_centroides
                self.clusters     = self.get_clusters()
                new_centroides  = self.new_centroide()
        else:
            raise Exception("Convergence criterion not valid.")
        if self.show:
            self.show_clusters()

    def Init_Centroide(self):
        centroides = []
        if self.init_method == 'random':
            for _ in range(1, self.k + 1):
                random_index = np.random.choice(self.data.shape[0])
                random_row = self.data[random_index]
                centroides.append(random_row)
        elif self.init_method == 'kmeans++':     
            # Get a random row from the matrix
            random_row = np.random.choice(self.data.shape[0])
            # Access the random row
            first_centroid = self.data[random_row]  
            centroides.append(first_centroid)
            
            for _ in range(1, self.k):
                global_distances = []
                for vc in self.data:
                    local_distances = []
                    for centroid in centroides:
                        local_distances.append(self.distance(vc, centroid))
                    global_distances.append(min(local_distances))
                
                # Convert distances to a numpy array and compute squared distances
                global_distances = np.array(global_distances)

                # Compute probabilities proportional to the squared distances
                probabilities = global_distances**2 / np.sum(global_distances**2)

                # Select the next centroid based on the probabilities
                centroid = self.data[np.random.choice(self.data.shape[0], p=probabilities)]

                centroides.append(centroid)
        else:
            raise Exception("Initialization method for centroids not valid.")
        return centroides

    def distance(self, v1, v2):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(v2-v1)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(v2 - v1))
        else:
            raise Exception("Distance metric not valid.")

    def get_clusters(self):
        self.labels = []
        cluster = {}
        for i in range(0, self.k):
            cluster[i] = []

        for _, X in enumerate(self.data):
            distances = []
            for k_index, centroid in enumerate(self.centroids):
                dis = self.distance(X, centroid)
                distances.append((k_index, dis))
            min_tuple = min(distances, key = lambda x: x[1])
            self.labels.append(min_tuple[0])
            cluster[min_tuple[0]].append(X)

        for key, value in cluster.items():
            cluster[key] = np.array(value)
        
        numbers = np.arange(0, self.k)
        self.correspondence = dict(zip(emotions, numbers))
        return cluster
    
    def new_centroide(self):
        new_centroide = []
        for key in self.clusters:
            mean_ = np.mean(self.clusters[key], axis = 0)
            new_centroide.append(mean_)

        return new_centroide
    
    def centroids_distance(self, new_centroides):
        distances = []
        for i in range(0, self.k):
            centroid_vc        = self.centroids[i]
            new_centroid_vc    = new_centroides[i]
            distances.append(self.distance(centroid_vc, new_centroid_vc))

        distances = np.array(distances)
        mean_distance = np.mean(distances, axis = 0)
        return mean_distance
    
    # def silhouette_score(self):
    #     distances = pairwise_distances(self.data)
    #     silhouette_avg = silhouette_score(distances, self.labels)
    #     # Create an empty similarity matrix
    #     n = self.data.shape[1]
    #     similarity_matrix = np.zeros((n, n))
    #     return similarity_matrix, distances, silhouette_avg
    
    def show_clusters(self):
        dim_reduction_methods_comparison(self.data, self.labels, self.centroids, self.k, self.path)

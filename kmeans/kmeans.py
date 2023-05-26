import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from Plotting.Dim_red_comparison import dim_reduction_methods_comparison
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(42)



emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


class Kmeans:
    def __init__(self, n_clusters, data_points, initialization_method, distance_metric, convergence_criterion, show, path):
        self.k                  = n_clusters
        self.data               = data_points
        self.init_method        = initialization_method
        self.distance_metric    = distance_metric
        self.convergence_crit   = convergence_criterion
        self.show               = show
        self.labels             = []
        self.clusters           = {}
        self.centroids          = []
        self.correspondence     = {}
        self.path               = path

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
        if self.show:
            self.plot_()
        if self.convergence_crit[0] == 'umbral':
            while(self.centroids_distance(new_centroides) > self.convergence_crit[1]):
                self.centroids  = new_centroides
                self.clusters     = self.get_clusters()
                new_centroides  = self.new_centroide()
                if self.show:
                    self.plot_()
        elif self.convergence_crit[0] == 'iteracion':
            pass
        else:
            raise Exception("Convergence criterion not valid.")

    def Init_Centroide(self):
        centroides = []
        len = self.data.shape[1]
        if self.init_method == 'random':
            # upper_bound = np.amax(self.data)
            # lower_bound = np.amin(self.data)
            for _ in range(1, self.k + 1):
                # x = (upper_bound - lower_bound) * np.random.rand(1, len) + lower_bound
                # print(x)
                random_index = np.random.choice(self.data.shape[0])
                random_row = self.data[random_index]
                centroides.append(random_row)
        elif self.init_method == 'kmeans++':     
            # Get a random row from the matrix
            random_row = np.random.choice(self.data.shape[0])
            # Access the random row
            first_centroid = self.data[random_row]  
            centroides.append(first_centroid)
            
            for _ in range(1, self.k + 1):
                # distances = [np.min(self.distance(x, centroid) for centroid in centroides) for x in self.data]
                global_distances = []
                for vc in self.data:
                    local_distances = []
                    for centroid in centroides:
                        local_distances.append(self.distance(vc, centroid))
                    # print(local_distances)
                    global_distances.append(min(local_distances))
                    # print(global_distances)
                    # input("continue")
                
                # Convert distances to a numpy array and compute squared distances
                global_distances = np.array(global_distances)
                # print(global_distances)
                # Compute probabilities proportional to the squared distances
                probabilities = global_distances**2 / np.sum(global_distances**2)
                # print(probabilities)
                # Select the next centroid based on the probabilities
                centroid = self.data[np.random.choice(self.data.shape[0], p=probabilities)]
                # print(centroid)
                centroides.append(centroid)
                # print(centroides)
                # input("next")
            # print(centroides)
            # input("continue")=
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
        # print("distancia promedio entre centroides (old vs. new): ", mean_distance)
        return mean_distance
    
    def silhouette_score(self):
        distances = pairwise_distances(self.data)
        silhouette_avg = silhouette_score(distances, self.labels)
        # Create an empty similarity matrix
        n = self.data.shape[1]
        similarity_matrix = np.zeros((n, n))
        return similarity_matrix, distances, silhouette_avg
    
    def plot_(self):
        # if self.path == '2_VC.csv':
            # print(self.labels)
        dim_reduction_methods_comparison(self.data, self.labels, self.centroids, self.k, self.path)
            # Create the animation
            # animation = FuncAnimation(fig, dim_reduction_methods_comparison, frames=num_frames, interval=200)

            # Save the animation as a GIF
            # animation.save('plot_animation.gif', writer='pillow')

        # pca = PCA(n_components = 2)
        # reduced_data = pca.fit_transform(self.data)
        # fig, ax = plt.subplots()

        # ax.clear()

        # ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c = self.labels)
        # ax.set_title("K-means Clustering")
        # plt.pause(0.5)

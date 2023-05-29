import numpy as np
from Plotting.Dim_red_comparison import dim_reduction_methods_comparison_dbscan


class DBSCAN:
    def __init__(self, data_points, eps, minPts, distance_metric, show):
        self.data               = data_points
        self.eps                = eps
        self.minPts             = minPts
        self.distance_metric    = distance_metric
        self.show               = show

    def distance(self, v1, v2):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(v2-v1)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(v2 - v1))
        else:
            raise Exception("Distance metric not valid.")

    def fit(self):
        self.labels             = np.zeros(len(self.data), dtype=int) - 1  # Initialize cluster labels as unclassified (-1)
        self.n_cluster          = 0

        for i, point in enumerate(self.data):
            if self.labels[i] != -1: # skip it
                continue

            neighbors = []
            for j, other_point in enumerate(self.data):
                dist = self.distance(point, other_point)
                if dist <= self.eps:
                    neighbors.append(j)
            
            if len(neighbors) < self.minPts:
                self.labels[i] = 0 # noise
                continue
            
            self.n_cluster += 1 # next cluster
            self.labels[i] = self.n_cluster # this point belongs to the cluster n_cluster
            neighbors_copy = neighbors
            neighbors_copy.remove(i)
            # neighbors = self.propagate(neighbors)

            # for neighbor in neighbors_copy:
            while len(neighbors_copy) > 0:
                neighbor = neighbors_copy.pop(0)
                if self.labels[neighbor] == 0:
                    self.labels[neighbor] = self.n_cluster
                if self.labels[neighbor] != -1:
                    continue
                
                new_neighbors = []
                for j, other_point in enumerate(self.data):
                    if self.distance(self.data[neighbor], other_point) <= self.eps:
                        new_neighbors.append(j)

                self.labels[neighbor] = self.n_cluster

                if len(new_neighbors) < self.minPts:
                    continue
                neighbors_copy.extend(new_neighbors)
                neighbors_copy = list(set(neighbors_copy))
                neighbors_copy.remove(neighbor)
        
        return self.labels, self.n_cluster


    def plot_(self, k , path):
        dim_reduction_methods_comparison_dbscan(self.data, self.labels, k , path)
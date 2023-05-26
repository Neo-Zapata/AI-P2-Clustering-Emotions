from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.cluster import KMeans
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
     

# Se carga el dataset

iris = load_iris()

X = iris.data[:,2:]
Y = iris.target.reshape(-1, 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


     

# Modelo
def gauss(X, clusters=3):
  # KMeans
  kmeans = KMeans(n_clusters=clusters, n_init="auto").fit(X)
  labels = kmeans.labels_
  centers = kmeans.cluster_centers_
  covs = []
  for i in range(clusters):
    covs.append(np.cov(X[labels==i].T))

  covs = np.array(covs)

  pi = np.array([(labels==i).sum() for i in range(clusters)]) / len(labels)

  return centers, covs, pi


def EM(X, centers, covs, pi):
  for epoch in range(100): # Se eligen 100 epochs
    if epoch % 50 == 0:
      print("EPOCH", epoch)


  # Cálculo de likelihood
  likelihood = np.zeros([X.shape[0], len(pi)])

  for i in range(len(pi)):
    multi_normal = stats.multivariate_normal(centers[i], covs[i])
    for j in range(X.shape[0]):
      likelihood[j][i] = multi_normal.pdf(X[j])
  
  for i in range(len(pi)):
    likelihood[:,i] *= pi[i]

  y = likelihood/likelihood.sum(1)[:,None]

  N = y.sum(0)
  for i in range(len(pi)):
    y_col = y[:, i].reshape(-1,1)
    centers[i] = np.sum(y_col*X, 0) / N[i]
    dif = X - centers[i]
    covs[i] = np.dot((y_col * dif).T, dif) 
    covs[i] /= N[i]

  pi = N/X.shape[0] 

  return centers, covs, pi, y

  
     

centers, covs, pi = gauss(X)
centers, covs, pi, y = EM(X, centers, covs, pi)

labels = np.argmax(y, 1)

print(labels.reshape(-1))
print(Y.reshape(-1))

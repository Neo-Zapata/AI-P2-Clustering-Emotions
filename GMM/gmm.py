import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class GMM:
    def __init__(self, n_clusters, max_epochs, convergence_threshold, data, file):
        self.n_clusters = n_clusters
        self.max_epochs = max_epochs
        self.convergence_threshold = convergence_threshold
        self.data = data
        self.file = file
        self.centers = None
        self.covs = None
        self.weights = None
        self.responsibilities = None

    def fit_(self, X):
        gmm = GaussianMixture(n_components=self.n_clusters)
        gmm.fit(X)
        self.centers = gmm.means_
        self.covs = gmm.covariances_
        self.weights = gmm.weights_

        # EM algorithm
        prev_log_likelihood = None
        for epoch in range(self.max_epochs):
            print("EPOCH", epoch)

            # E-step
            log_likelihood, responsibilities = self._e_step(X)

            # Check convergence
            if prev_log_likelihood is not None and np.abs(log_likelihood - prev_log_likelihood) < self.convergence_threshold:
                break

            # M-step
            self._m_step(X, responsibilities)

            prev_log_likelihood = log_likelihood

    def _e_step(self, X):
        likelihood = np.zeros([X.shape[0], self.n_clusters])
        for i in range(self.n_clusters):
            multi_normal = multivariate_normal(self.centers[i], self.covs[i])
            likelihood[:, i] = multi_normal.pdf(X)

        likelihood *= self.weights

        responsibilities = likelihood / likelihood.sum(axis=1, keepdims=True)
        responsibilities = np.nan_to_num(responsibilities, nan=1e-8)

        log_likelihood = np.log(likelihood.sum(axis=1)).sum()

        self.responsibilities = responsibilities
        
        return log_likelihood, responsibilities

    def _m_step(self, X, responsibilities):
        N = responsibilities.sum(axis=0)

        for i in range(self.n_clusters):
            resp_col = responsibilities[:, i].reshape(-1, 1)
            self.centers[i] = np.sum(resp_col * X, axis=0) / N[i]
            dif = X - self.centers[i]
            self.covs[i] = np.dot((resp_col * dif).T, dif) / N[i]

        self.weights = N / X.shape[0]

    def plot(self, X):
        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=self.responsibilities.argmax(axis=1))

        # Plot cluster centers
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='red', marker='x', label='Cluster Centers')

        # Plot ellipses for covariance matrices
        for i in range(self.n_clusters):
            cov = self.covs[i]
            center = self.centers[i]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width = 2 * np.sqrt(2 * eigenvalues[0])
            height = 2 * np.sqrt(2 * eigenvalues[1])

            ellipse = Ellipse(center, width, height, angle, edgecolor='r', facecolor='none')
            plt.gca().add_patch(ellipse)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('GMM Clustering Results')
        plt.legend()
        plt.show()

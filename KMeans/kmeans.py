import numpy as np

class KMeans:
    def __init__(self, X, k, init_means = None, max_iters=100):
        self.X = X
        self.k = k
        self.max_iters = max_iters
        if init_means:
            assert len(init_means) == self.k, "The number of initial means doesn't match with the number of clusters"
            self.means = np.array(init_means)
        else:
            self.means = np.random.randint(low=0, high=256, size=(3, 3))
        self.fit()

    def assign_clusters(self):
        # compute Euclidean distance of each point to each mean and assign cluster
        distances = np.linalg.norm(self.X[:, np.newaxis] - self.means, axis=2)
        self.labels = np.argmin(distances, axis=1)

    def update_means(self):
        new_means = []
        for i in range(self.k):
            cluster_points = self.X[self.labels == i]
            if len(cluster_points) == 0:
                # if a cluster gets no points, reinitialize randomly
                new_mean = np.random.randint(low=0, high=256, size=(self.X.shape[1],))
            else:
                new_mean = np.mean(cluster_points, axis=0)
            new_means.append(new_mean)
        self.means = np.array(new_means)

    def fit(self, tol=1e-4):
        for _ in range(self.max_iters):
            prev_means = self.means.copy()
            self.assign_clusters()
            self.update_means()
            if np.all(np.abs(self.means - prev_means) < tol):
                break

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.means, axis=2)
        return np.argmin(distances, axis=1)
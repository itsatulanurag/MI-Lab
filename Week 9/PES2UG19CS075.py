import numpy as np


class KMeansClustering:

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def e_step(self, data):
        dist = np.zeros((data.shape[0], self.n_cluster))
        for i in range(self.n_cluster):
            temp = np.square(data-self.centroids[i, :])
            dist[:, i] = np.sqrt(np.sum(temp, axis=1))
        return np.argmin(dist, axis=1)

    def m_step(self, data, cluster_assgn):
        x = self.n_cluster
        for i in range(x):
            self.centroids[i, :] = np.mean(data[cluster_assgn == i, :], axis=0)

    def evaluate(self, data):
        dist = []
        count = 0
        x = len(data)
        y = len(self.centroids)
        for i in range(x):
            for j in range(y):
                dist.append(np.square(self.centroids[j]-data[i]))
        dist = np.sum(dist, axis=1)
        for i in dist:
            count += i
        return count

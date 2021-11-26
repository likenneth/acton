import numpy as numpy
from sklearn.cluster import KMeans
import numpy as np

class Clusterer:
    def __init__(self, TIMES, K, TOL):
        # cannot use custom distance
        self.K = K
        self.num_init = TIMES
        self.kmeans = KMeans(n_clusters=self.K, n_init=self.num_init, verbose=False, tol=TOL)
        # tol: Relative tolerance with regards to Frobenius norm of the difference 
        # in the cluster centers of two consecutive iterations to declare convergence

    def fit(self, x):  # x: (num_sample, num_feat)
        new_kmeans = self.kmeans.fit(x)
        self.kmeans = new_kmeans
        # after sort, we sort the best centroids in front
        score_container = []
        for i in range(self.K):
            score = self.kmeans.score(x[self.kmeans.labels_ == i])  # the bigger the better
            score_container.append(score)
        indices = np.argsort(np.array(score_container))[::-1]
        self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_[indices]

    def get_assignment(self, x):
        # x: (num_sample, num_feat)
        # returns the centroids with same shape
        idx = self.kmeans.predict(x)
        # centroids = self.kmeans.cluster_centers_[idx]
        return idx

    def get_centroids(self, ):
        for idx in range(self.K):
            yield self.kmeans.cluster_centers_[idx]

def get_best_clusterer(nodes, times, argument_dict):
    c = Clusterer(TIMES=times, K=argument_dict["K"], TOL=argument_dict["TOL"])
    c.fit(nodes)
    return c
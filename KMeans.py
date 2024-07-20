# The trained model is the locations of the centroids
# when a new observation comes in, whichever centriod that's closest to it determines its cluster label
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, epoch=100):
        self.n_clusters = n_clusters
        self.epoch = epoch
        self.X_train = None
        self.n_train = None
        self.centroids = None

    def fit(self, X_train):
        self.X_train = X_train
        self.n_train = X_train.shape[0]
        # initialize with random centroids: cannot exceed training boundaries
        min_, max_ = np.min(self.X_train, axis=0), np.max(self.X_train, axis=0)
        # generate n_clusters points with the correct dimensions
        centroids = [np.random.uniform(min_, max_) for _ in range(self.n_clusters)]
        # train for given number of epochs
        for i in range(self.epoch):
            centroids = self.update_centroids(centroids)
        self.centroids = centroids

    def euclidean(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def update_centroids(self, centroids):
        # store cluster labels of training data
        labels = np.zeros(self.n_train)
        # assign each training data to closest centroid
        for i in range(self.n_train):
            dists = [self.euclidean(self.X_train[i], centroid) for centroid in centroids]
            # index of closest centroid is cluster label
            labels[i] = np.argmin(dists)
        # update centroids by averaging points in given cluster
        for i in range(self.n_clusters):
            # all points assigned to given cluster
            points = self.X_train[np.array(labels)==i]
            centroids[i] = points.mean(axis=0)
        return centroids

    def predict(self, X_test):
        y_preds = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            p = X_test[i]
            dists = [self.euclidean(p, centroid) for centroid in self.centroids]
            y_preds[i] = np.argmin(dists)
        return y_preds

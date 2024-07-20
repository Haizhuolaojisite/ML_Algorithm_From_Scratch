# knn is an non-parametric method
# it makes no assumptions about population distribution
# knn makes predictions by memorizing the training data
# when a new observation comes in, we compute its distance (e.g. Euclidean, Manhattan)
# from all the training data to find its k nearest neighbors
# For classification, take a majority vote on K neighbors' labels to predict the label of new observation
# For regression, we can take the average target value of k neighbors to make a prediction
from scipy.stats import mode
import numpy as np


class KNeighborsClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.n_train = X_train.shape[0]


    def predict(self, X_test):
        self.X_test = X_test
        self.n_test = X_test.shape[0]
        y_pred = np.zeros(self.n_test)
        for i in range(self.n_test):
            # find k nearest neighbors of given point
            p = self.X_test[i]
            neighbors = np.zeros(self.n_neighbors)
            neighbors = self.find_neighbors(p)
            # take a majority vote on the final label
            y_pred[i] = mode(neighbors)[0][0]
        return y_pred

    def euclidean(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def find_neighbors(self, p):
        # compute distance btw given point and all train points
        distances = np.zeros(self.n_train)
        for i in range(self.n_train):
            distance = self.euclidean(p, self.X_train[i])
            distances[i] = distance
        # sort training labels by distance ascending
        _, y_train_sorted = zip(*sorted(zip(distances, self.y_train)))
        return y_train_sorted[:self.n_neighbors]
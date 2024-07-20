from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train[:5])
# [[ 0.83998789 -0.05678968]
#  [ 1.02682187 -1.28080657]
#  [-0.34227589  1.13144745]
#  [ 1.03010341 -1.38991625]
#  [ 1.31529441 -1.49607792]]

class LogisticRegression:
    def __init__(self, lr=0.01, epoch=10):
        self.lr = lr
        self.epoch = epoch
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_obs, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for i in range(self.epoch):
            y_pred = self.predict(X)
            dw = -(1/n_obs) * (np.dot(X.T, (y - y_pred)))
            db = -(1/n_obs) * (np.sum(y - y_pred))
            self.w = self.w - (self.lr * dw)
            self.b = self.b - (self.lr * db)

    def predict(self, X):
        z = X.dot(self.w) + self.b
        # apply sigmoid function to transform z into prob [0, 1]
        p = self.sigmoid(z)
        y_pred = np.where(p < 0.5, 0, 1)
        return y_pred

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

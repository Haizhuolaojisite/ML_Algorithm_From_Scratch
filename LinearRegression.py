import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
# from LinearRegression import LinearRegression
n_samples = 1000
n_features = 10
X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       n_informative=5, noise=10, random_state=42)
X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=.2, random_state=1)

class LinearRegression:
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        '''
        Gradient Descent with MSE as its cost function to update weights
        :param X: observations
        :param y: dependent variable
        '''
        # initialize weights and bias
        n_observations = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        tol = 1e-5
        prev_loss = 0
        for i in range(self.epochs):
            y_pred = self.predict(X)
            dw = (-2 / n_observations) * (np.dot(X.T, (y_pred - y)))
            db = (-2 / n_observations) * (np.sum(y_pred - y))
            # adjust the step size using a learning rate parameter
            # The cross-validation can be used to find an ideal learning rate
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

            current_loss = np.mean(np.square(y_pred - y))

            if abs(current_loss - prev_loss) < tol:
                break

            prev_loss = current_loss

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


model = LinearRegression(lr=0.01)
model.fit(X_fit, y_fit)
train_predictions = model.predict(X_fit)


def mse(y, predictions):
    return np.mean(np.square(y - predictions))


def rmse(y, predictions):
    return np.sqrt(mse(y, predictions))


print(mse(y_fit, train_predictions))

y_pred_train = model.predict(X_fit)
y_pred_val = model.predict(X_val)

fig = plt.figure(figsize=(6,4))
plt.scatter(y_val, y_pred_val, color="green", s=5)
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linewidth=2)
plt.xlabel("True values")
plt.ylabel("Predicted validation values")
plt.show()
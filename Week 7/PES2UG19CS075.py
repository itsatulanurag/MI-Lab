import numpy as np
from numpy.random.mtrand import sample
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:

    def __init__(self, n_stumps=20):
        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        self.alphas = []
        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):
            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)
            self.stumps.append(st)
            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)
        return self

    def stump_error(self, y, y_pred, sample_weights):
        error = 0
        for i in range(len(y)):
            if(y[i] != y_pred[i]):
                error += sample_weights[i]
        return error

    def compute_alpha(self, error):
        eps = 1e-9
        if(error == 0):
            error = eps
        a = 0.5*np.log((1-error)/error)
        return a

    def update_weights(self, y, y_pred, sample_weights, alpha):
        new_weight = [0]*len(y)
        for i in range(len(y)):
            if(y[i] != y_pred[i]):
                new_weight[i] = sample_weights[i]*np.exp(alpha)
            else:
                new_weight[i] = sample_weights[i]*np.exp(-1*alpha)
        new_samples = new_weight/sum(new_weight)
        return new_samples

    def predict(self, X):
        pred = np.array([self.stumps[i].predict(X)
                        for i in range(self.n_stumps)])
        return np.sign(pred[0])

    def evaluate(self, X, y):
        pred = self.predict(X)
        correct = (pred == y)

        accuracy = np.mean(correct) * 100
        return accuracy

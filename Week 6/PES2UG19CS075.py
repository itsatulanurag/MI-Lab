from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
import pandas as pd
import numpy as np


class SVM:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        data = pd.read_csv(self.dataset_path)
        self.X = data.iloc[:, 0:-1]
        self.y = data.iloc[:, -1]

    def solve(self):
        scalar = 'scalar', StandardScaler()
        power = 'power', PowerTransformer()
        quantile = 'quantile', QuantileTransformer()
        result = Pipeline(
            [scalar, power, quantile, ('svc', SVC(kernel='rbf', gamma=1))])
        result.fit(self.X, self.y)
        return(result)

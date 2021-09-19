import numpy as np
from scipy.spatial import distance
from math import *
import copy


class KNN:
    def __init__(self, k_neigh, weighted=False, p=2):
        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        self.data = data
        self.target = target.astype(np.int64)
        return self

    def find_distance(self, x):
        pass

    def k_neighbours(self, x):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass

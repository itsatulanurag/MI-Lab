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
<<<<<<< HEAD
        result = []
        for vec1 in x:
            temp_arr = []
            for vec2 in self.data:
                d = distance.minkowski(vec1, vec2, self.p)
                temp_arr.append(d)
            result.append(temp_arr)
        return result
=======
        pass
>>>>>>> 8e11fe46291b9ba5a0033e3add28d204700b8a74

    def k_neighbours(self, x):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass

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
        result = []
        for vec1 in x:
            temp_arr = []
            for vec2 in self.data:
                d = distance.minkowski(vec1, vec2, self.p)
                temp_arr.append(d)
            result.append(temp_arr)
        return result

    def k_neighbours(self, x):
        arr = KNN.find_distance(self, x)
        temp_arr = copy.deepcopy(arr)
        for set in arr:
            set.sort()
        res = []
        ind = []
        for i in range(len(x)):
            res.append(arr[i][slice(self.k_neigh)])
        res = [res]
        # for i in range(self.k_neigh):
        #     for j in range(temp_arr):
        #         if(temp_arr[i][j])
        # print(j)
        # print(temp_arr)
        return res

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass

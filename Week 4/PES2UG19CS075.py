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
            ind_temp = []
            for j in range(self.k_neigh):
                for k in range(len(temp_arr[i])):
                    if(res[i][j] == temp_arr[i][k]):
                        ind_temp.append(k)
            ind.append(ind_temp)
        return (res, ind)

    def predict(self, x):
        neighbors = self.k_neighbours(x)[1]
        result = []
        for i in range(len(neighbors)):
            temp_arr = {}
            for j in range(len(neighbors[i])):
                if self.target[neighbors[i][j]] in temp_arr:
                    temp_arr[self.target[neighbors[i][j]]] += 1
                else:
                    temp_arr[self.target[neighbors[i][j]]] = 1
            max_val = 0
            pred = 0
            for i in range(min(temp_arr), max(temp_arr)+1):
                if temp_arr[i] > max_val:
                    max_val = temp_arr[i]
                    pred = i
            result.append(pred)
        return result

    def evaluate(self, x, y):
        pred_arr = KNN.predict(self, x)
        count = np.sum(pred_arr == y)
        result = 100*(count/len(x))
        return result

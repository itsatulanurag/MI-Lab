import numpy as np
from numpy.core.numeric import identity
import pandas as pd


def create_numpy_ones_array(shape):
    array = None
    array = np.ones(shape)
    return array


def create_numpy_zeros_array(shape):
    array = None
    array = np.zeros(shape)
    return array


def create_identity_numpy_array(shape):
    array = None
    array = identity(shape)
    return array


def matrix_cofactor(array):
    rows, cols = array.shape
    arr = np.zeros(array.shape)
    for r in range(rows):
        for c in range(cols):
            minor = array[np.array(list(range(r))+list(range(r+1, rows)))[:, np.newaxis],
                          np.array(list(range(c))+list(range(c+1, cols)))]
            arr[r, c] = (-1)**(r+c)*np.linalg.det(minor)
    return arr


def f1(X1, coef1, X2, coef2, seed1, seed2, seed3, shape1, shape2):
    ans = None
    np.random.seed(seed1)
    w1 = np.random.rand(*shape1)
    np.random.seed(seed2)
    w2 = np.random.rand(*shape2)
    x = X1.shape
    y = X2.shape
    if(x[0] == shape1[1] and y[0] == shape2[1]) and (shape1[0] == shape2[0] and x[1] == y[1]):
        p1 = np.dot(w1, X1*coef1)
        p2 = np.dot(w2, X2**coef2)
        np.random.seed(seed3)
        b = np.random.randint(10, size=p1.shape)
        ans = p1+p2+b
    else:
        ans = -1
    return ans


def fill_with_mode(filename, column):
    df = pd.read_csv(filename)
    df[column].fillna(df[column].mode()[0], inplace=True)
    return df


def fill_with_group_average(df, group, column):
    df1 = df.groupby(group)
    df[column].fillna(df1[column].transform('mean'), inplace=True)
    return df


def get_rows_greater_than_avg(df, column):
    df = df[df[column] > df[column].mean()]
    return df

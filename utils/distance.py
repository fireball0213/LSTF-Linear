import numpy as np


def euclidean(A, B):
    #检查A，B的维度是否为1
    if A.ndim == 1:
        return np.linalg.norm(B - A)
    return np.linalg.norm(B - A, axis=1)


#一个函数，曼哈顿距离
def manhattan(A, B):
    if A.ndim == 1:
        return np.sum(np.abs(B - A))
    return np.sum(np.abs(B - A), axis=1)

#一个函数，切比雪夫距离
def chebyshev(A, B):
    if A.ndim == 1:
        return np.max(np.abs(B - A))
    return np.max(np.abs(B - A), axis=1)




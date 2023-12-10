import numpy as np


def euclidean(A, B):
    #     #检查A，B的维度是否为1
    #     if A.ndim == 1:
    #         return np.linalg.norm(B - A)
    #     return np.linalg.norm(B - A, axis=1)
    # 如果 A 是一维数组，将其扩展为二维数组（具有单个行）
    if B.ndim == 1 and A.ndim == 1:
        return np.linalg.norm(B - A)
    if A.ndim == 1:
        A = A[np.newaxis, :]
    # 计算 A 中每个元素与 B 中元素之间的欧几里得距离
    return np.linalg.norm(B - A, axis=1)

#一个函数，曼哈顿距离
def manhattan(A, B):
    # if A.ndim == 1:
    #     return np.sum(np.abs(B - A))
    # return np.sum(np.abs(B - A), axis=1)
    if B.ndim == 1 and A.ndim == 1:
        return np.sum(np.abs(B - A))
    if A.ndim == 1:
        A = A[np.newaxis, :]
    return np.sum(np.abs(B - A), axis=1)

#一个函数，切比雪夫距离
def chebyshev(A, B):
    # if A.ndim == 1:
    #     return np.max(np.abs(B - A))
    # return np.max(np.abs(B - A), axis=1)
    if B.ndim == 1 and A.ndim == 1:
        return np.max(np.abs(B - A))
    if A.ndim == 1:
        A = A[np.newaxis, :]
    return np.max(np.abs(B - A), axis=1)




import numpy as np
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from scipy.spatial.distance import pdist, squareform
import time

from scipy.spatial.distance import cdist

def get_distance(args):
    distance_dict = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'chebyshev': chebyshev,
        'mahalanobis': mahalanobis_distance,
        'weighted_euclidean': weighted_euclidean,
    }
    return distance_dict[args.distance]

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

def preprocess_ts2(ts2):
    """
    预处理 ts2，计算协方差矩阵的逆。
    """
    n_variables = ts2.shape[2]
    combined_ts = ts2.reshape(-1, n_variables)
    covariance_matrix = np.cov(combined_ts.T)
    inv_covmat = inv(covariance_matrix)
    return inv_covmat

def preprocess_inv(x):
    """
    预处理 x，计算协方差矩阵的逆。
    """
    covariance_matrix = np.cov(x.T)
    inv_covmat = inv(covariance_matrix)
    return inv_covmat
#处理多变量序列的距离度量函数，确保距离函数能够比较两个多变量时间序列，并返回一个标量距离值，使用马氏距离考虑数据集中各变量的协方差
def mahalanobis_distance(ts1, ts2,inv_covmat):
    """
    ts1: 第一个时间序列, shape = (1,seq_len, n_variables)
    ts2: 第二个时间序列, shape = (num_samples, seq_len, n_variables)
    inv_covmat: ts2预处理计算出的协方差矩阵的逆，是(n_variables, n_variables)的对角矩阵
    return: 马氏距离, shape = (num_samples,)
    """
    # 广播 ts1 来匹配 ts2 的形状，然后计算差值
    delta = ts1 - ts2  # 结果的形状是 (num_samples, seq_len, n_variables)

    # 应用协方差矩阵的逆
    temp = np.matmul(delta, inv_covmat)  # 最快
    # temp = np.einsum('ijk,kl->ijl', delta, inv_covmat)
    # temp = np.dot(delta, inv_covmat)

    # 计算每个样本的马氏距离
    distances = np.mean(np.sqrt(np.einsum('ijk,ijk->ij', temp, delta)), axis=1)#最快
    # distances = np.sqrt(np.sum(temp * delta, axis=(1, 2)))

    return distances

#处理多变量序列的距离度量函数，确保距离函数能够比较两个多变量时间序列，并返回一个标量距离值，使用加权欧几里得距离可以根据变量的重要性对距离度量进行加权
def weighted_euclidean(ts1, ts2, weights=None):
    """
    ts1: 第一个时间序列, shape = (1,seq_len, n_variables)
    ts2: 第二个时间序列, shape = (num_samples, seq_len, n_variables)
    return: shape = (num_samples,)
    """
    # 如果没有指定权重，那么所有变量的权重都设置为1
    if weights is None:
        weights = np.ones(ts1.shape[-1])
    # Ensure that weights are a numpy array and reshape to align with the variables
    weights = np.array(weights).reshape(1, 1, -1)
    weighted_diff = weights * (ts1 - ts2)
    distances = np.linalg.norm(weighted_diff, axis=(1, 2))
    return distances



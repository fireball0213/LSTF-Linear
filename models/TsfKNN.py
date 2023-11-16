import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.seasonal import STL
from models.base import MLForecastModel
from utils.distance import euclidean, manhattan, chebyshev
#显示进度
from tqdm import tqdm
#可视化
from matplotlib import pyplot as plt

class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        if args.distance == 'euclidean':
            self.distance = euclidean
        elif args.distance == 'manhattan':
            self.distance = manhattan
        elif args.distance == 'chebyshev':
            self.distance = chebyshev
        self.decompose = args.decompose#是否考虑趋势和季节性
        self.seasonal = args.seasonal#季节性的值
        self.msas = args.msas
        self.x_stl=None
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, -1]

    def _stl_modified_distance(self, x, y_series):
        # 这里假设x是一个时间序列，y_series是多个时间序列的集合
        x_stl = STL(x, period=52).fit()
        self.x_stl=x_stl

        distances = []
        for i in range(y_series.shape[0]):
            y = np.copy(y_series[i])
            y_stl = STL(y, period=52).fit()#,seasonal=self.seasonal
            dist = self.distance(x_stl.trend, y_stl.trend) + self.distance(x_stl.seasonal, y_stl.seasonal)
            distances.append(dist)
            #
        return np.array(distances)


    def _search(self, x, X_s, seq_len, pred_len):
        #X_s是滑动窗口后的训练集，X_s的维度为(num_samples, seq_len+pred_len)
        #找到训练集中与x最相似的k个时间序列，然后对这k个时间序列的后pred_len个值求均值，作为预测值
        if self.msas == 'MIMO':
            if self.decompose:
                distances = self._stl_modified_distance(x, X_s[:, :seq_len])
            else:
                distances = self.distance(x, X_s[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:]# 获取这k个最近邻时间序列的预测部分
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)# 计算这些最近邻时间序列预测的平均值
            return x_fore
        elif self.msas == 'recursive':
            if self.decompose:
                distances = self._stl_modified_distance(x, X_s[:, :seq_len])
            else:
                distances = self.distance(x, X_s[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len].reshape((-1, 1))# 获取这k个最近邻时间序列的下一个时间点的值
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)# 计算这些最近邻时间序列下一个时间点的平均值
            x_new = np.concatenate((x[:, 1:], x_fore), axis=1)# 更新x，准备下一次递归预测
            if pred_len == 1:
                return x_fore
            else:
                return np.concatenate((x_fore, self._search(x_new, X_s, seq_len, pred_len - 1)), axis=1)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        seq_len = X.shape[1]
        X_s = sliding_window_view(self.X, seq_len + pred_len)
        #显示进度
        for x in tqdm(X):
            #打印进度
            #print('predicting the {}th sample'.format(len(fore)+1))
            x = np.expand_dims(x, axis=0)
            x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore

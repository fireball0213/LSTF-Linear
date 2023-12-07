import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.seasonal import STL
from models.base import MLForecastModel
from utils.distance import euclidean, manhattan, chebyshev
from tqdm import tqdm
from dataset.data_visualizer import plot_STL
from sklearn.linear_model import LinearRegression
from datasketch import MinHash, MinHashLSH
from lshashpy3 import LSHash
import matplotlib.pyplot as plt
from utils.metrics import mse, mae, mape, smape, mase
import time

class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        if args.distance == 'euclidean':
            self.distance = euclidean
        elif args.distance == 'manhattan':
            self.distance = manhattan
        elif args.distance == 'chebyshev':
            self.distance = chebyshev
        self.decompose = args.decompose  # 是否考虑趋势和季节性
        self.trend = args.trend
        self.msas = args.msas
        self.period = args.period  # 季节性的值
        self.approximate_knn = args.approximate_knn  # 是否使用近似knn
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.hash_size = args.hash_size
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, -1]
        self.X_slide = sliding_window_view(self.X, self.seq_len + self.pred_len)
        if self.decompose:
            self.X_stl = STL(self.X, period=self.period).fit()  # 对整个序列进行STL分解
            # plot_STL(self.X_stl,400)
            subseries = sliding_window_view(self.X_stl.trend, self.seq_len + self.pred_len)
            self.trend_model = LinearRegression()
            if self.trend == 'AR':
                trend_X = subseries[:, :self.seq_len]
                trend_y = subseries[:, self.seq_len:]
                self.trend_model.fit(trend_X, trend_y)
        if self.approximate_knn:
            self.lsh_model = LSHash(hash_size=self.hash_size, input_dim=self.seq_len)
            for i, d in enumerate(self.X_slide):
                point = d[:self.seq_len]
                self.lsh_model.index(point, extra_data=i)

    def _search(self, x, X_slide, seq_len, pred_len):
        # 找到训练集中与x最相似的k个时间序列，然后对这k个时间序列的后pred_len个值求均值，作为预测值
        if self.approximate_knn == False:
            distances = self.distance(x, X_slide[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
        else:
            result = self.lsh_model.query(x.ravel(), num_results=self.k)
            indices_of_smallest_k = [res[0][1] for res in result]

        if self.msas == 'MIMO':
            neighbor_fore = X_slide[indices_of_smallest_k, seq_len:]  # 获取这k个最近邻时间序列的预测部分
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)  # 计算这些最近邻时间序列预测的平均值
            return x_fore
        elif self.msas == 'recursive':
            neighbor_fore = X_slide[indices_of_smallest_k, seq_len].reshape((-1, 1))  # 获取这k个最近邻时间序列的下一个时间点的值
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)  # 计算这些最近邻时间序列下一个时间点的平均值
            x_new = np.concatenate((x[:, 1:], x_fore), axis=1)  # 更新x，准备下一次递归预测
            if pred_len == 1:
                return x_fore
            else:
                return np.concatenate((x_fore, self._search(x_new, X_slide, seq_len, pred_len - 1)), axis=1)

    def STL_search(self,x_stl_origin, x_stl_trend,x_stl_seasonal , x_stl_resid, seq_len, pred_len):
        #在STL分解后的序列上进行搜索
        # X_s_origin = sliding_window_view(self.X_stl.observed, seq_len + pred_len)
        X_s_trend = sliding_window_view(self.X_stl.trend, seq_len + pred_len)
        X_s_seasonal = sliding_window_view(self.X_stl.seasonal, seq_len + pred_len)
        X_s_resid = sliding_window_view(self.X_stl.resid, seq_len + pred_len)
        '''
        优化后的代码，快了3倍，不再需要_stl_modified_distance函数，但需要distance函数支持向量化
        '''
        if self.approximate_knn == False and self.msas == 'MIMO':
            if self.trend == 'STL':
                distances = self.distance(x_stl_trend+x_stl_seasonal ,X_s_trend[:, :seq_len]+X_s_seasonal[:, :seq_len])
                # distances = self.distance(x_stl_origin, X_s_origin[:, :seq_len])
                indices_of_smallest_k = np.argsort(distances)[:self.k]
                # neighbor_fore = X_s_origin[indices_of_smallest_k, seq_len:]#等价于不适用STL
                neighbor_fore = X_s_trend[indices_of_smallest_k, seq_len:] + X_s_seasonal[indices_of_smallest_k,seq_len:]
            elif self.trend == 't_s':
                distance_t = self.distance(x_stl_trend, X_s_trend[:, :seq_len])
                distance_s = self.distance(x_stl_seasonal, X_s_seasonal[:, :seq_len])
                indices_of_smallest_k_t = np.argsort(distance_t)[:self.k]
                indices_of_smallest_k_s = np.argsort(distance_s)[:self.k]
                neighbor_fore = X_s_trend[indices_of_smallest_k_t, seq_len:] + X_s_seasonal[indices_of_smallest_k_s,seq_len:]
            else:
                # distances=self.distance(x_stl_seasonal+ x_stl_resid, X_s_seasonal[:, :seq_len]+X_s_resid[:, :seq_len])#使用季节性和残差计算距离
                distances = self.distance(x_stl_seasonal, X_s_seasonal[:, :seq_len])  # 使用季节性计算距离
                indices_of_smallest_k = np.argsort(distances)[:self.k]
                neighbor_fore = X_s_resid[indices_of_smallest_k, seq_len:] + X_s_seasonal[indices_of_smallest_k,seq_len:]  # 使用季节性+残差作为预测
                # neighbor_fore = X_s_seasonal[indices_of_smallest_k, seq_len:]  # 使用季节性作为预测

        else:
            '''
            不能只用季节性，因为太小了，lsh没见过这么小的数据，会匹配不到
            总会出现搜不到的情况，注意处理，使用非模糊knn
            '''
            result = self.lsh_model.query((x_stl_trend + x_stl_seasonal+x_stl_resid).ravel(), num_results=self.k)

            if len(result) <= 1 :
                #搜不到时，不使用LSH
                distances = self.distance(x_stl_seasonal, X_s_seasonal[:, :seq_len])  # 使用季节性计算距离
                indices_of_smallest_k = np.argsort(distances)[:self.k]
            else:
                indices_of_smallest_k = [res[0][1] for res in result]
            neighbor_fore = X_s_resid[indices_of_smallest_k, seq_len:] + X_s_seasonal[indices_of_smallest_k,seq_len:]  # 使用季节性+残差作为预测
            # neighbor_fore = X_s_seasonal[indices_of_smallest_k, seq_len:]  # 使用季节性作为预测
        x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
        return x_fore


    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        for x in tqdm(X):
            x = np.expand_dims(x, axis=0)
            x_stl = STL(x[0], period=self.period).fit()

            if self.decompose:
                # 传入STL分解后的序列进行搜索
                x_fore = self.STL_search(x_stl.observed,x_stl.trend,x_stl.seasonal,x_stl.resid, self.seq_len, pred_len)
                if self.trend == 'AR':
                    x_stl_trend = self.trend_model.predict(x_stl.trend.reshape((1, -1))[:, -self.seq_len:])
                    x_fore += x_stl_trend.ravel()
                elif self.trend == 'plain':
                    self.trend_model.fit(np.arange(self.seq_len).reshape((-1, 1)),x_stl.trend.reshape((-1, 1))[:, -self.seq_len:])
                    x_stl_trend = self.trend_model.predict(np.arange(self.seq_len, pred_len + self.seq_len).reshape((-1, 1)))
                    x_fore += x_stl_trend.ravel()
            else:
                x_fore = self._search(x, self.X_slide, self.seq_len, pred_len)
            fore.append(x_fore)

        fore = np.array(fore).reshape((-1, pred_len))
        return fore

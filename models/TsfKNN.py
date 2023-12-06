import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.seasonal import STL
from models.base import MLForecastModel
from utils.distance import euclidean, manhattan, chebyshev
#显示进度
from tqdm import tqdm
from dataset.data_visualizer import plot_STL
from sklearn.linear_model import LinearRegression
from datasketch import MinHash, MinHashLSH
from lshashpy3 import LSHash
import matplotlib.pyplot as plt
from utils.metrics import mse, mae, mape, smape, mase

Num_perm=1024
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
        self.msas = args.msas
        self.period = args.period#季节性的值
        self.approximate_knn= args.approximate_knn#是否使用近似knn
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.hash_size=args.hash_size
        if self.approximate_knn:
            self.lsh_model = LSHash(hash_size=self.hash_size, input_dim=self.seq_len)
        super().__init__()


    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, -1]
        self.X_slide=sliding_window_view(self.X, self.seq_len )
        self.X_s=sliding_window_view(self.X, self.seq_len + self.pred_len)
        if self.decompose:
            self.X_stl = STL(self.X, period=self.period).fit()  # 对整个序列进行STL分解
            # plot_STL(self.X_stl,2000)
            subseries = sliding_window_view(self.X_stl.trend, self.seq_len + self.pred_len)
            self.trend_model = LinearRegression()
            trend_X = subseries[:, :self.seq_len]
            trend_y = subseries[:, self.seq_len:]
            self.trend_model.fit(trend_X, trend_y)
        if self.approximate_knn:
            for i, d in enumerate(self.X_s):
                point = np.array([d[:self.seq_len]]) if d.ndim == 0 else d[:self.seq_len]
                self.lsh_model.index(point, extra_data=i)



    def _lsh_search(self, query, num_results):
        result = self.lsh_model.query(query.ravel(), num_results=num_results)
        result_lst=[res[0][1] for res in result]
        return result_lst


    def _stl_modified_distance(self, x_component, y_components_series):
        # x_component 是单个时间序列的 STL 分解结果（趋势或季节性组件）
        # y_components_series 是原数据滑动窗口的 STL 分解结果（趋势或季节性组件）
        distances = []
        for y_component in y_components_series:
            # 计算 x_component 与 y_component 之间的距离
            dist = self.distance(x_component, y_component)
            distances.append(dist)
        return np.array(distances)



    def _search(self, x, X_s, seq_len, pred_len):
        # 找到训练集中与x最相似的k个时间序列，然后对这k个时间序列的后pred_len个值求均值，作为预测值
        if self.approximate_knn==False:
            distances = self.distance(x, X_s[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
        else:
            indices_of_smallest_k = self._lsh_search(x, self.k)
            # print(f"indices_of_smallest_k: {indices_of_smallest_k}")

        if self.msas == 'MIMO':
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:]# 获取这k个最近邻时间序列的预测部分
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)# 计算这些最近邻时间序列预测的平均值
            return x_fore
        elif self.msas == 'recursive':
            neighbor_fore = X_s[indices_of_smallest_k, seq_len].reshape((-1, 1))# 获取这k个最近邻时间序列的下一个时间点的值
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)# 计算这些最近邻时间序列下一个时间点的平均值
            x_new = np.concatenate((x[:, 1:], x_fore), axis=1)# 更新x，准备下一次递归预测
            if pred_len == 1:
                return x_fore
            else:
                return np.concatenate((x_fore, self._search(x_new, X_s, seq_len, pred_len - 1)), axis=1)



    def STL_search(self, x_stl_seasonal,x_stl_resid, X_s_seasonal, X_s_resid, seq_len, pred_len):
        # 假设 x_stl 是单个时间序列的 STL 分解结果
        # X_s_trend 和 X_s_seasonal 是训练数据集的趋势和季节性组件的窗口
        if self.approximate_knn == False:
            # distances=self._stl_modified_distance(x_stl.seasonal+ x_stl.resid, X_s_seasonal[:, :seq_len]+X_s_resid[:, :seq_len])#使用季节性和残差计算距离
            distances = self._stl_modified_distance(x_stl_seasonal, X_s_seasonal[:, :seq_len])  # 使用季节性计算距离
            indices_of_smallest_k = np.argsort(distances)[:self.k]
        else:
            indices_of_smallest_k = self._lsh_search(x_stl_seasonal, self.k)

        if self.msas == 'MIMO':
            neighbor_fore = X_s_resid[indices_of_smallest_k, seq_len:] + X_s_seasonal[indices_of_smallest_k, seq_len:]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore
        elif self.msas == 'recursive':
            neighbor_fore = X_s_resid[indices_of_smallest_k, seq_len] + X_s_seasonal[indices_of_smallest_k, seq_len]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            x_new_seasonal = np.concatenate((x_stl_seasonal[ 1:], x_fore), axis=0)
            x_new_resid = np.concatenate((x_stl_resid[1:], x_fore), axis=0)
            if pred_len == 1:
                return x_fore.reshape(1, -1)
            else:
                return np.concatenate((x_fore.reshape(1, -1), self.STL_search(x_new_seasonal,x_new_resid, X_s_seasonal, X_s_resid, seq_len, pred_len - 1)), axis=1)


    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        seq_len = X.shape[1]

        #显示进度
        for x in tqdm(X):
            x = np.expand_dims(x, axis=0)
            x_stl = STL(x[0], period=self.period).fit()
            # 传入STL分解后的序列进行搜索
            if self.decompose:
                # X_s_trend = sliding_window_view(self.X_stl.trend, seq_len + pred_len)
                X_s_seasonal = sliding_window_view(self.X_stl.seasonal, seq_len + pred_len)
                X_s_resid = sliding_window_view(self.X_stl.resid, seq_len + pred_len)
                x_fore = self.STL_search(x_stl.seasonal,x_stl.resid,  X_s_seasonal,X_s_resid, seq_len, pred_len)
                x_stl_trend=self.trend_model.predict(x_stl.trend.reshape((1,-1))[:,-seq_len:])
                x_fore+=x_stl_trend.ravel()
            else:
                x_fore = self._search(x, self.X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore



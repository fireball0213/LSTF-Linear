import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.seasonal import STL
from models.base import MLForecastModel
from utils.distance import euclidean, manhattan, chebyshev
#显示进度
from tqdm import tqdm
from dataset.data_visualizer import plot_STL
from sklearn import linear_model
#LinearRegression
from sklearn.linear_model import LinearRegression

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
        # self.seasonal = args.seasonal#季节性的值
        self.msas = args.msas
        self.period = args.period
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, -1]
        if self.decompose:
            self.X_stl = STL(self.X, period=self.period).fit()  # 对整个序列进行STL分解
            plot_STL(self.X_stl,2000)



    def _stl_modified_distance(self, x_component, y_components_series):
        # x_component 是单个时间序列的 STL 分解结果的趋势或季节性组件
        # y_components_series 是多个时间序列的 STL 分解结果的趋势或季节性组件的集合

        distances = []
        for y_component in y_components_series:
            # 计算 x_component 与 y_component 之间的距离
            dist = self.distance(x_component, y_component)
            distances.append(dist)
        return np.array(distances)

    def _search(self, x, X_s, seq_len, pred_len):
        #X_s是滑动窗口后的训练集，X_s的维度为(num_samples, seq_len+pred_len)
        #找到训练集中与x最相似的k个时间序列，然后对这k个时间序列的后pred_len个值求均值，作为预测值
        if self.msas == 'MIMO':
            distances = self.distance(x, X_s[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:]# 获取这k个最近邻时间序列的预测部分
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)# 计算这些最近邻时间序列预测的平均值
            return x_fore
        elif self.msas == 'recursive':
            distances = self.distance(x, X_s[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len].reshape((-1, 1))# 获取这k个最近邻时间序列的下一个时间点的值
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)# 计算这些最近邻时间序列下一个时间点的平均值
            x_new = np.concatenate((x[:, 1:], x_fore), axis=1)# 更新x，准备下一次递归预测
            if pred_len == 1:
                return x_fore
            else:
                return np.concatenate((x_fore, self._search(x_new, X_s, seq_len, pred_len - 1)), axis=1)

    def STL_search(self, x_stl, X_s_seasonal, X_s_resid, seq_len, pred_len):
        # 假设 x_stl 是单个时间序列的 STL 分解结果
        # X_s_trend 和 X_s_seasonal 是训练数据集的趋势和季节性组件的窗口

        if self.msas == 'MIMO':
            # 分别计算与 x_stl 的趋势和季节性组件的距离
            # distances_trend = self._stl_modified_distance(x_stl.trend, X_s_trend[:, :seq_len])
            # distances_seasonal = self._stl_modified_distance(x_stl.seasonal, X_s_seasonal[:, :seq_len])
            # distances_resid = self._stl_modified_distance(x_stl.resid, X_s_resid[:, :seq_len])
            # distances = distances_trend + distances_seasonal
            # distances = distances_seasonal + distances_resid

            #使用季节性和残差计算距离
            # distances=self._stl_modified_distance(x_stl.seasonal+ x_stl.resid, X_s_seasonal[:, :seq_len]+X_s_resid[:, :seq_len])
            distances = self._stl_modified_distance(x_stl.seasonal, X_s_seasonal[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s_resid[indices_of_smallest_k, seq_len:] + X_s_seasonal[indices_of_smallest_k, seq_len:]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore
        # elif self.msas == 'recursive':
        #     distances_trend = self._stl_modified_distance(x_stl.trend, X_s_trend[:, :seq_len])
        #     distances_seasonal = self._stl_modified_distance(x_stl.seasonal, X_s_seasonal[:, :seq_len])
        #     distances = distances_trend + distances_seasonal
        #
        #     indices_of_smallest_k = np.argsort(distances)[:self.k]
        #     neighbor_fore = X_s_trend[indices_of_smallest_k, seq_len] + X_s_seasonal[indices_of_smallest_k, seq_len]
        #     x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
        #     x_stl_fore = STL(x_fore[0], period=self.period).fit()
        #     x_stl_new = STL(np.concatenate((x_stl.trend[:, 1:], x_stl_fore.trend), axis=1), period=self.period).fit()
        #     if pred_len == 1:
        #         return x_fore
        #     else:
        #         return np.concatenate((x_fore, self.STL_search(x_stl_new, X_s_trend, X_s_seasonal, seq_len, pred_len - 1)), axis=1)
    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        seq_len = X.shape[1]
        X_s = sliding_window_view(self.X, seq_len + pred_len)
        # 对self.X_stl的trend和seasonal分别使用sliding_window_view
        if self.decompose:
            X_s_trend = sliding_window_view(self.X_stl.trend, seq_len + pred_len)
            X_s_seasonal = sliding_window_view(self.X_stl.seasonal, seq_len + pred_len)
            X_s_resid = sliding_window_view(self.X_stl.resid, seq_len + pred_len)
        #显示进度
        for x in tqdm(X):
            #打印进度
            #print('predicting the {}th sample'.format(len(fore)+1))
            x = np.expand_dims(x, axis=0)
            x_stl = STL(x[0], period=self.period).fit()
            # 传入STL分解后的序列进行搜索
            if self.decompose:
                x_fore = self.STL_search(x_stl,  X_s_seasonal,X_s_resid, seq_len, pred_len)
                #使用线性回归模型，预测x_stl.trend的未来pred_len个时间点的值
                model=LinearRegression()
                model.fit(np.arange(seq_len).reshape((-1,1)),x_stl.trend.reshape((-1,1))[:,-seq_len:])
                x_stl_trend=model.predict(np.arange(seq_len,pred_len+seq_len).reshape((-1,1))).reshape((1,-1))
                x_fore+=x_stl_trend
            else:
                x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore



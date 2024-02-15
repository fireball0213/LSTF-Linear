import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.seasonal import STL
from models.base import MLForecastModel
from utils.distance import euclidean, manhattan, chebyshev,get_distance,preprocess_ts2,preprocess_inv
from utils.decomposition import get_decompose
from tqdm import tqdm
from dataset.data_visualizer import plot_STL,plot_decompose,plot_slide
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from utils.metrics import mse, mae, mape, smape, mase
import time

class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.dia_func= args.distance
        self.distance = get_distance(args)
        self.decompose = get_decompose(args)  # 是否考虑趋势和季节性
        self.period = args.period  # 季节性的值
        self.trend = args.trend #趋势的预测方法
        self.distance_dim = args.distance_dim  # 是否是多变量距离
        self.weighted = args.weighted
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        if self.distance_dim == 'OT':
            # self.X = X[0, :, -1]
            self.X = X[:, 0]
            self.X_slide = sliding_window_view(self.X, self.seq_len + self.pred_len)
            self.X_slide_seq = self.X_slide[:, :self.seq_len]
            if self.decompose is not None:
                self.X_trend,self.X_seasonal,self.X_resid = self.decompose(self.X, self.period)  # 使用不同方法对整个序列进行季节性趋势分解
                # plot_decompose(self.X,self.X_trend,self.X_seasonal,self.X_resid,0,300,model=self.decompose.__name__)

                self.trend_model = LinearRegression()
                self.seasonal_model = LinearRegression()
                if self.trend == 'AR' or self.trend == 'AR_AR':
                    subsidies_trend = sliding_window_view(self.X_trend, self.seq_len + self.pred_len)
                    trend_X = subsidies_trend[:, :self.seq_len]
                    trend_y = subsidies_trend[:, self.seq_len:]
                    self.trend_model.fit(trend_X, trend_y)
                if self.trend == 'AR_AR':
                    subsidies_seasonal = sliding_window_view(self.X_seasonal, self.seq_len + self.pred_len)
                    seasonal_X = subsidies_seasonal[:, :self.seq_len]
                    seasonal_y = subsidies_seasonal[:, self.seq_len:]
                    self.seasonal_model.fit(seasonal_X, seasonal_y)
        elif self.distance_dim=='multi':
            self.X = X[0, :, :]
            self.X_slide = sliding_window_view(self.X, (self.seq_len + self.pred_len,1))
            self.X_slide = self.X_slide.transpose(0,2,1,3)[:,:,:,0]
            self.X_slide_seq = self.X_slide[:, :self.seq_len]
            if self.dia_func == 'mahalanobis':
                self.inv_covmat = preprocess_inv(self.X)#使用sliding前数据，更合理
            elif self.dia_func == 'weighted_euclidean':
                weights = np.array(self.weighted)
                '''
                权重稀疏时，可以加速计算
                '''
                self.nonzero_w_index = weights != 0
                self.nonzero_weights = weights[self.nonzero_w_index]

    def _search(self, x, pred_len):
        # 找到训练集中与x最相似的k个时间序列，然后对这k个时间序列的后pred_len个值求均值，作为预测值
        if self.dia_func == 'mahalanobis':
            distances = self.distance(x, self.X_slide_seq, self.inv_covmat)
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = self.X_slide[indices_of_smallest_k, self.seq_len:,-1]
        elif self.dia_func == 'weighted_euclidean':
            distances = self.distance(x[..., self.nonzero_w_index], self.X_slide_seq[..., self.nonzero_w_index],
                                      weights=self.nonzero_weights)
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = self.X_slide[indices_of_smallest_k, self.seq_len:, -1]
        else:#'OT'
            distances = self.distance(x, self.X_slide_seq)
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = self.X_slide[indices_of_smallest_k, self.seq_len:]  # 获取这k个最近邻时间序列的预测部分
        x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)  # 计算这些最近邻时间序列预测的平均值
        return x_fore

    def decompose_search(self, x_stl_trend,x_stl_seasonal , pred_len):
        #在STL分解后的序列上进行搜索

        X_s_trend = sliding_window_view(self.X_trend, self.seq_len + pred_len)
        X_s_seasonal = sliding_window_view(self.X_seasonal, self.seq_len + pred_len)
        X_s_resid = sliding_window_view(self.X_resid, self.seq_len + pred_len)
        if self.trend == 't_plus_s':
            distances = self.distance(x_stl_trend+x_stl_seasonal ,X_s_trend[:, :self.seq_len]+X_s_seasonal[:, :self.seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore_trend = X_s_trend[indices_of_smallest_k, self.seq_len:]
            neighbor_fore_seasonal = X_s_seasonal[indices_of_smallest_k,self.seq_len:]
            trend_fore = np.mean(neighbor_fore_trend, axis=0, keepdims=True)
            seasonal_fore = np.mean(neighbor_fore_seasonal, axis=0, keepdims=True)
        elif self.trend == 't_s':
            distance_t = self.distance(x_stl_trend, X_s_trend[:, :self.seq_len])
            distance_s = self.distance(x_stl_seasonal, X_s_seasonal[:, :self.seq_len])
            indices_of_smallest_k_t = np.argsort(distance_t)[:self.k]
            indices_of_smallest_k_s = np.argsort(distance_s)[:self.k]
            neighbor_fore_trend = X_s_trend[indices_of_smallest_k_t, self.seq_len:]
            neighbor_fore_seasonal = X_s_seasonal[indices_of_smallest_k_s, self.seq_len:]
            trend_fore = np.mean(neighbor_fore_trend, axis=0, keepdims=True)
            seasonal_fore = np.mean(neighbor_fore_seasonal, axis=0, keepdims=True)
        elif self.trend == 'AR':
            distances = self.distance(x_stl_seasonal, X_s_seasonal[:, :self.seq_len])  # 使用季节性计算距离
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s_seasonal[indices_of_smallest_k, self.seq_len:]  # 使用季节性作为预测
            trend_fore = self.trend_model.predict(x_stl_trend.reshape((1, -1))[:, -self.seq_len:])
            seasonal_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
        elif self.trend == 'AR_AR':
            x=x_stl_trend.reshape((1, -1))[:, -self.seq_len:]
            trend_fore = self.trend_model.predict(x)
            seasonal_fore = self.seasonal_model.predict(x_stl_seasonal.reshape((1, -1))[:, -self.seq_len:])
        else:
            distances = self.distance(x_stl_seasonal, X_s_seasonal[:, :self.seq_len])  # 使用季节性计算距离
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s_seasonal[indices_of_smallest_k, self.seq_len:]  # 使用季节性作为预测
            self.trend_model.fit(np.arange(self.seq_len).reshape((-1, 1)), x_stl_trend.reshape((-1, 1))[:, -self.seq_len:])
            trend_fore = self.trend_model.predict(np.arange(self.seq_len, pred_len + self.seq_len).reshape((-1, 1))).reshape(1,-1)
            seasonal_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
        return trend_fore,seasonal_fore


    def _forecast(self, X) -> np.ndarray:
        fore = []
        if self.decompose is not None:
            #还原测试序列，仅限单变量时使用
            testX=np.concatenate((X[:,0],X[-1,1:]),axis=0)
            testX_trend, testX_seasonal, testX_resid = self.decompose(testX, self.period)
            testX_trend_slide = sliding_window_view(testX_trend, self.seq_len )
            testX_seasonal_slide = sliding_window_view(testX_seasonal, self.seq_len )
            testX_resid_slide = sliding_window_view(testX_resid, self.seq_len )
            for i,x in enumerate(tqdm(X)):
                # x_trend,x_seasonal,x_resid = self.decompose(x[0],self.period)
                '''
                优化为先还原测试序列，再统一STL分解。更合理也更快
                '''
                x_trend=testX_trend_slide[i]
                x_seasonal=testX_seasonal_slide[i]
                x_resid=testX_resid_slide[i]
                # plot_decompose(x, x_trend, x_seasonal, x_resid, self.seq_len, model=self.decompose.__name__)
                trend_fore,seasonal_fore = self.decompose_search(x_trend,x_seasonal,  self.pred_len)
                x_fore = trend_fore+seasonal_fore

                #可画图单独查看分量预测效果
                # x_fore = trend_fore
                # x_fore = seasonal_fore

                fore.append(x_fore)
        else:
            for i, x in enumerate(tqdm(X)):
                x = np.expand_dims(x, axis=0)
                x_fore = self._search(x,  self.pred_len)
                fore.append(x_fore)

        fore = np.array(fore).reshape((-1, self.pred_len))
        return fore

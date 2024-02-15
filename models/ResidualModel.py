from models.TsfKNN import TsfKNN
from models.DLinear import DLinear
from models.baselines import Autoregression
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
from models.TsfKNN import TsfKNN
from models.baselines import ZeroForecast, MeanForecast
from models.baselines import Autoregression,ExponentialMovingAverage,DoubleExponentialSmoothing,LastValueForecast
from models.DLinear import Linear_NN, NLinear, DLinear
from models.ThetaMethod import ThetaMethodForecast
from utils.transforms import IdentityTransform, Normalization, Standardization,MeanNormalization,BoxCox,FourierTransform
from torch.utils.data import DataLoader
from trainer import MLTrainer
from dataset.dataset import get_dataset
from dataset.ETT_data import Dataset_ETT_hour
from dataset.data_visualizer import data_visualize,plot_forecast,plot_day_forecast,plot_random_forecast
import matplotlib.pyplot as plt
import os
import argparse
import random
import torch
import numpy as np
from models.ARIMA import ARIMAForecast
def get_model(args,key=None):
    model_dict = {
        'ZeroForecast': ZeroForecast,
        'MeanForecast': MeanForecast,
        'LastValueForecast': LastValueForecast,
        'Autoregression': Autoregression,
        'ExponentialMovingAverage': ExponentialMovingAverage,
        'DoubleExponentialSmoothing': DoubleExponentialSmoothing,
        'TsfKNN': TsfKNN,
        'Linear_NN':Linear_NN,
        'NLinear' : NLinear,
        'DLinear': DLinear,
        'ARIMA': ARIMAForecast,
        'ThetaMethod':ThetaMethodForecast,
        'ResidualModel':ResidualModel,
    }
    if key is not None:
        return model_dict[key](args)
    return model_dict[args.model](args)


class ResidualModel(MLForecastModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.trend_model = get_model(args, args.trend_model)
        self.seasonal_model = get_model(args, args.seasonal_model)
        self.decompose_method = get_decompose(args)
        self.period = args.period
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len


    def fit(self, X):
        if len(X.shape) == 3:
            self.X = X[0, :, -1]
        else:
            self.X = X.ravel()
        self.X_slide = sliding_window_view(self.X, self.seq_len + self.pred_len)
        # self.X_slide_seq = self.X_slide[:, :self.seq_len]
        if self.decompose_method is not None:
            self.X_trend, self.X_seasonal, self.X_resid = self.decompose_method(self.X, self.period)  # 使用不同方法对整个序列进行季节性趋势分解
            # plot_decompose(self.X,self.X_trend,self.X_seasonal,self.X_resid,0,300,model=self.decompose.__name__)

            self.trend_model.fit(self.X_trend)
            self.seasonal_model.fit(self.X_seasonal)

    def forecast(self, X):
        fore = []
        if self.decompose_method is not None:
            # 还原测试序列，仅限单变量时使用
            testX = np.concatenate((X[:, 0], X[-1, 1:]), axis=0)

            testX_trend, testX_seasonal, testX_resid = self.decompose_method(testX, self.period)

            testX_trend_slide = sliding_window_view(testX_trend, self.seq_len)
            testX_seasonal_slide = sliding_window_view(testX_seasonal, self.seq_len)
            testX_resid_slide = sliding_window_view(testX_resid, self.seq_len)

            # for i, x in enumerate(tqdm(X)):
            #     '''
            #     优化为先还原测试序列，再统一STL分解。更合理也更快
            #     '''
            #     x_trend = testX_trend_slide[i]
            #     x_seasonal = testX_seasonal_slide[i]
            #     x_resid = testX_resid_slide[i]
            #     # plot_decompose(x, x_trend, x_seasonal, x_resid, self.seq_len, model=self.decompose.__name__)
            #
            #     trend_fore = self.trend_model.forecast(x_trend.reshape((1, -1))[:, -self.seq_len:])
            #     seasonal_fore = self.seasonal_model.forecast(x_seasonal.reshape((1, -1))[:, -self.seq_len:])
            #
            #     x_fore = trend_fore + seasonal_fore
            #
            #     # 可画图单独查看分量预测效果
            #     # x_fore = trend_fore
            #     # x_fore = seasonal_fore
            #
            #     fore.append(x_fore)

            trend_fore = self.trend_model.forecast(testX_trend_slide)
            seasonal_fore = self.seasonal_model.forecast(testX_seasonal_slide)
            fore = trend_fore + seasonal_fore

        fore = np.array(fore).reshape((-1, self.pred_len))
        return fore



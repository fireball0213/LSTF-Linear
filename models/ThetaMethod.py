from models.base import MLForecastModel
import tqdm
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.linear_model import LinearRegression
class ThetaMethodForecast():
    def __init__(self, args):
        super().__init__()
        self.seasonal_periods = args.period  # 季节性周期
        self.seq_len = args.seq_len  # 输入长度
        self.pred_len = args.pred_len  # 预测长度
        self.channels = args.channels
        self.theta_list = args.theta_list  # Theta系数列表,default=[0,0.5,1, 2]


    def fit(self, X: np.ndarray,X_trend,X_seasonal):
        # 在这个方法中，无需训练数据， 因为我们将在forecast方法中为每个序列拟合模型
        #但选取最佳的Theta系数（从一系列候选系数中）需要历史数据
        #X:(num_input,channels)
        pass

    def forecast_series(self, args):
        series, seasonal, coefficient = args
        # 应用Theta系数进行趋势调整
        first_diff = np.diff(series, n=1)
        second_diff = np.diff(series, n=2) * coefficient
        adjusted_first_diff = np.concatenate(([first_diff[0]], np.cumsum(second_diff) + first_diff[0]))
        adjusted_series = np.cumsum(np.concatenate(([series[0]], adjusted_first_diff))) + series[0]

        model = ExponentialSmoothing(adjusted_series, trend='add', seasonal=None, initialization_method="estimated")
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=self.pred_len)
        return forecast + seasonal[-self.pred_len:]  # 将季节性成分添加到趋势预测中

    def forecast(self, X: np.ndarray, X_trend: np.ndarray, X_seasonal: np.ndarray) -> np.ndarray:
        forecasts = np.zeros((X.shape[0], self.pred_len, self.channels))
        for j in range(self.channels):
            args_list = [(X_trend[i, :, j], X_seasonal[i, :, j], coef) for i in range(X.shape[0]) for coef in self.theta_list]

            with ProcessPoolExecutor() as executor:
                forecast_results = list(tqdm.tqdm(executor.map(self.forecast_series, args_list), total=len(args_list)))

            # Reshape results and compute the mean across theta coefficients for each series
            forecast_results = np.array(forecast_results).reshape(X_trend.shape[0], len(self.theta_list), self.pred_len)
            forecasts[:, :, j] = forecast_results.mean(axis=1)  # Take mean across theta coefficients

        return forecasts
#
# class ThetaMethodForecast():
#     def __init__(self, args):
#         super().__init__()
#         self.seasonal_periods = args.period  # 季节性周期
#         self.seq_len = args.seq_len  # 输入长度
#         self.pred_len = args.pred_len  # 预测长度
#         self.channels = args.channels
#         self.theta_list = args.theta_list  # Theta系数列表,default=[0,0.5,1, 2]
#         self.models = [LinearRegression() for _ in range(self.channels)]
#
#     def apply_theta_adjustment(self, series, coefficient):
#         # 应用Theta系数进行趋势调整
#         first_diff = np.diff(series, n=1)
#         second_diff = np.diff(series, n=2) * coefficient
#         adjusted_first_diff = np.concatenate(([first_diff[0]], np.cumsum(second_diff) + first_diff[0]))
#         adjusted_series = np.cumsum(np.concatenate(([series[0]], adjusted_first_diff))) + series[0]
#         return adjusted_series
#     def fit(self, X, X_trend, X_seasonal):
#         #仅利用趋势分量进行训练
#         for i in range(self.channels):
#             adjusted_series = self.apply_theta_adjustment(X_trend[:, i], self.theta_list)
#             subseries = sliding_window_view(adjusted_series, self.seq_len + self.pred_len)
#             train_X = subseries[:, :self.seq_len]
#             train_Y = subseries[:, self.seq_len:]
#             self.models[i].fit(train_X, train_Y)
#
#     def forecast(self, X, X_trend, X_seasonal):
#         forecasts = np.zeros((X.shape[0], self.pred_len, self.channels))
#         for j in range(self.channels):
#             adjusted_series=[self.apply_theta_adjustment(X_trend[i,:, j], self.theta_list) for i in range(X.shape[0])]
#             predictions = self.models[j].predict(adjusted_series)
#             forecasts[:, :, j] = predictions + X_seasonal[:, -self.pred_len:, j]
#         return forecasts
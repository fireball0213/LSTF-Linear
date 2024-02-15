import numpy as np
from models.base import MLForecastModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tqdm
# class ThetaMethodForecast(MLForecastModel):
#     def __init__(self, args):
#         super().__init__()
#         self.theta = args.theta  # Theta参数
#         self.seasonal_periods = args.period  # 季节性周期
#
#     def _fit(self, X: np.ndarray) -> None:
#         # Theta方法的拟合部分
#         self.models = []
#         X = X[:, :, -1]  # 只使用最后一个特征/通道
#         # for i in range(X.shape[2]):  # 遍历每个特征/通道
#         #     model = ExponentialSmoothing(X[:, :, i].squeeze(), trend='add',
#         #                                  seasonal='add', seasonal_periods=self.seasonal_periods)
#         #     fitted_model = model.fit()
#         #     self.models.append(fitted_model)
#         model = ExponentialSmoothing(X.squeeze(), trend='add',
#                                      seasonal='add', seasonal_periods=self.seasonal_periods)
#         fitted_model = model.fit()
#         self.models.append(fitted_model)
#
#     def _forecast(self, X: np.ndarray, pred_len: int) -> np.ndarray:
#         # Theta方法的预测部分
#         # forecasts = []
#         # for model in self.models:
#         #     forecast = model.forecast(steps=pred_len)
#         #     forecasts.append(forecast)
#         # forecasts = np.stack(forecasts, axis=-1)
#         # 初始化一个空的预测数组
#         forecasts = np.zeros((X.shape[0], pred_len))
#
#         # 循环遍历每个序列进行预测
#         for i in tqdm.tqdm(range(X.shape[0])):
#             # 从模型中获取预测
#             forecast = self.models[0].forecast(steps=pred_len)
#             forecasts[i] = forecast
#         return forecasts
#
#     def update_args(self, args):
#         # 更新参数的方法，以便在需要时调整Theta方法的配置
#         self.theta = args.theta
#         self.seasonal_periods = args.period
#         self.trend_method = args.trend_method
#         self.seasonal_method = args.seasonal_method
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from joblib import Parallel, delayed

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class ThetaMethodForecast(MLForecastModel):
    def __init__(self, args):
        super().__init__()
        self.seasonal_periods = args.period  # 季节性周期
        self.pred_len = args.pred_len  # 预测长度

    def _fit(self, X: np.ndarray):
        # 在这个方法中，我们不执行实际的拟合过程
        # 因为我们将在_forecast方法中为每个序列拟合模型
        pass

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        # 为X中的每个时间序列进行预测
        # 注意：这里假设X的形状是 (num_series, num_timesteps, num_features)
        # 并且我们只关注最后一个特征/通道的时间序列
        forecasts = np.zeros((X.shape[0], self.pred_len))
        for i in tqdm.tqdm(range(X.shape[0])):
            series = X[i, :]  # 选择最后一个特征/通道
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=self.seasonal_periods)
            fitted_model = model.fit()
            forecasts[i, :] = fitted_model.forecast(steps=self.pred_len)
        return forecasts

    def update_args(self, args):
        # 更新参数的方法，以便在需要时调整Theta方法的配置
        self.seasonal_periods = args.period

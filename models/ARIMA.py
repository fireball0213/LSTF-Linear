import numpy as np
from models.base import MLForecastModel
from pmdarima.arima import auto_arima
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tqdm
# class ARIMAForecast(MLForecastModel):
#     def __init__(self, args) -> None:
#         super().__init__()
#         self.args = args
#         self.model = None
#         self.model_path = 'sarimax_model_1.pkl'  # 模型保存路径
#
#     def _fit(self, X: np.ndarray) -> None:
#         # 确保输入数据是一维数组
#         X = X[:, :, -1].ravel()
#         # self.model = auto_arima(X, seasonal=True, m=self.args.period, suppress_warnings=True)
#         # self.model = auto_arima(X, seasonal=True, m=self.args.period, suppress_warnings=True,
#         #                         stepwise=True, trace=True, n_jobs=4,)#start_p=1, start_q=1, max_p=3, max_q=3, start_P=1, start_Q=1, max_P=2, max_Q=2
#         self.model = auto_arima(
#             X,
#             seasonal=True,
#             m=self.args.period,
#             suppress_warnings=True,
#             stepwise=True,  # 使用逐步搜索
#             trace=True,
#             n_jobs=4,  # 并行执行的作业数
#             start_p=0, start_q=0,
#             max_p=5, max_q=5,  # 扩大p和q的搜索范围
#             start_P=0, start_Q=0,
#             max_P=3, max_Q=3,  # 扩大P和Q的搜索范围
#             max_d=2, max_D=2,  # 调整差分阶数的上限
#             information_criterion='aic',  # 选择信息准则，可选'aic'、'bic'、'hqic'
#             d=None, D=None,  # 允许auto_arima自动确定最优的差分阶数
#         )
#
#         print(self.model.summary())
#
#         # 将找到的最佳模型参数用于SARIMAX
#         order = self.model.order
#         seasonal_order = self.model.seasonal_order
#         self.model = SARIMAX(X, order=order, seasonal_order=seasonal_order).fit(disp=False)
#         # self.model = SARIMAX(X, order=(1, 1, 1), seasonal_order=(2, 0, 0, 24)).fit(disp=False)
#         # 保存模型
#         # joblib.dump(self.model, self.model_path)
#         # self.fitted = True  # 标记模型为已训练
#
#     # def _load_model(self) -> None:
#     #     # 从磁盘加载模型
#     #     self.model = joblib.load(self.model_path)
#     #     self.fitted = True
#
#     def _forecast(self, X: np.ndarray, pred_len: int) -> np.ndarray:
#         # 确保模型已加载
#         # if not self.fitted:
#         #     self._load_model()
#
#         # 初始化一个空的预测数组
#         forecasts = np.zeros((X.shape[0], pred_len))
#
#         # 循环遍历每个序列进行预测
#         for i in tqdm.tqdm(range(X.shape[0])):
#         # for i in range(X.shape[0]):
#             # 假设每个序列可以独立使用同一个模型进行预测
#             # 这里的实现需要根据你的模型具体情况进行调整
#             # 例如，你可能需要重新拟合模型，或者调整模型以接受不同长度的时间序列
#             single_forecast = self.model.forecast(steps=pred_len)
#             forecasts[i, :] = single_forecast
#
#         return forecasts
#
#     def forecast(self, X: np.ndarray, pred_len: int) -> np.ndarray:
#         # 重写forecast方法以符合基类定义
#         return super().forecast(X, pred_len)


from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import tqdm
from models.base import MLForecastModel

from concurrent.futures import ProcessPoolExecutor

# class ARIMAForecast():
#     def __init__(self, args) -> None:
#         super().__init__()
#         self.args = args
#         self.pred_len = args.pred_len
#
#     def fit(self, X: np.ndarray,X_trend,X_seasonal):
#         # 在这个方法中，我们不执行实际的拟合过程
#         # 因为我们将在_forecast方法中为每个序列拟合模型
#         pass
#     def forecast(self, X: np.ndarray,X_trend,X_seasonal) -> np.ndarray:
#         forecasts = np.zeros((X.shape[0], self.pred_len))
#
#         # 循环遍历每个序列进行独立模型的拟合和预测
#         for i in tqdm.tqdm(range(X.shape[0])):
#             # 提取当前序列
#             current_sequence = X[i, :]  # 假设最后一个特征是我们关心的时间序列数据
#             # 创建并拟合模型
#             model = SARIMAX(current_sequence, order=(1, 1, 1), seasonal_order=(2, 0, 0, self.args.period)).fit(disp=False)
#             # 进行预测
#             single_forecast = model.forecast(steps=self.pred_len)
#             forecasts[i, :] = single_forecast
#
#         return forecasts



class ARIMAForecast():
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len

    def fit(self, X: np.ndarray,X_trend,X_seasonal):
        # 在这个方法中，我们不执行实际的拟合过程
        # 因为我们将在_forecast方法中为每个序列拟合模型
        pass
    def fit_forecast_single_sequence(self, sequence_data):
        current_sequence, pred_len, period = sequence_data
        model = SARIMAX(current_sequence, order=(1, 1, 1), seasonal_order=(2, 0, 0, period)).fit(disp=False)
        single_forecast = model.forecast(steps=pred_len)
        return single_forecast

    def forecast(self, X: np.ndarray,X_trend,X_seasonal) -> np.ndarray:
        sequences_data = [(X[i, :], self.pred_len, self.args.period) for i in range(X.shape[0])]
        forecasts = np.zeros((X.shape[0], self.pred_len))

        with ProcessPoolExecutor() as executor:
            results = list(
                tqdm.tqdm(executor.map(self.fit_forecast_single_sequence, sequences_data), total=len(sequences_data)))

        for i, single_forecast in enumerate(results):
            forecasts[i, :] = single_forecast

        return forecasts


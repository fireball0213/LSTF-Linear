import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from models.base import MLForecastModel
#导入线性回归模型
from sklearn import linear_model

class ZeroForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len))


class MeanForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(X, axis=-1).reshape(X.shape[0], 1)
        return np.repeat(mean, pred_len, axis=1)

#一个类，直接使用上一个时间步的值作为预测值
class LastValueForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        a=X[:, -1].reshape(X.shape[0], 1)
        b=np.repeat(a, pred_len, axis=1)
        return np.repeat(X[:, -1].reshape(X.shape[0], 1), pred_len, axis=1)


#一个线性回归的类，实现target列的自回归预测
class Autoregression(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.seq_len=args.seq_len
        self.pred_len = args.pred_len
        self.model = linear_model.LinearRegression()
        self.msas= args.msas

    def _fit(self, X: np.ndarray) -> None:
        X_target=X[:, :, -1]
        if self.msas=="recursive":
            subseries = np.concatenate(([sliding_window_view(v, self.seq_len + 1) for v in X_target]))#单输出迭代预测
        elif self.msas=="MIMO":
            subseries = np.concatenate([sliding_window_view(v, self.seq_len + self.pred_len) for v in X_target])#MIMO预测
        train_X = subseries[:, :self.seq_len]
        train_Y = subseries[:, self.seq_len:]
        self.model.fit(train_X, train_Y)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        if self.msas=="recursive":#单输出迭代预测
        #生成输出长度为pred_len的预测值，self.model只能输出1个值，所以需要将预测值一个一个的添加到X中，滑动窗口预测
        #X的维度为(num_samples, seq_len),预测值输出的维度为(num_samples, pred_len)
        # 初始化预测数组
            predictions = np.empty((X.shape[0], pred_len))
            for i in range(pred_len):
                # 使用当前窗口进行预测
                current_window = X[:, -self.seq_len:]
                pred = self.model.predict(current_window)
                # 更新预测结果
                predictions[:, i] = pred.reshape(-1)
                # 更新输入数据以包含最新预测
                X = np.concatenate((X, pred.reshape(-1, 1)), axis=1)
            return predictions
        elif self.msas == "MIMO":
            #MIMO预测
            assert pred_len == self.pred_len, "预测长度与模型训练时的长度不一致"
            # 使用模型直接进行预测
            current_window = X[:, -self.seq_len:]
            predictions = self.model.predict(current_window)
            return predictions

#一个指数滑动平均的类
class ExponentialMovingAverage(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.alpha=args.alpha
        self.seq_len=args.seq_len
        self.pred_len = args.pred_len
        self.msas= args.msas


    def _fit(self, X: np.ndarray) -> None:
        X_target = X[:, :, -1]
        #计算EMA，再使用EMA作为特征进行预测
        #EMA计算公式为：EMA[i]=alpha*data[i]+(1-alpha)*EMA[i-1]
        #初始化EMA
        EMA=np.zeros((X_target.shape[0], X_target.shape[1]))
        EMA[:, 0]=X_target[:, 0]
        for i in range(1, X_target.shape[1]):
            EMA[:, i]=self.alpha*X_target[:, i]+(1-self.alpha)*EMA[:, i-1]
        #使用EMA作为特征进行预测
        if self.msas=="recursive":
            subseries = np.concatenate(([sliding_window_view(v, self.seq_len + 1) for v in EMA]))
        elif self.msas == "MIMO":
            subseries = np.concatenate(([sliding_window_view(v, self.seq_len + self.pred_len) for v in EMA]))

        train_X = subseries[:, :self.seq_len]
        train_Y = subseries[:, self.seq_len:]
        self.model = linear_model.LinearRegression()
        self.model.fit(train_X, train_Y)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        if self.msas=="recursive":
        #单输出迭代预测
        #生成输出长度为pred_len的预测值，self.model只能输出1个值，所以需要将预测值一个一个的添加到X中，滑动窗口预测
        #X的维度为(num_samples, seq_len),预测值输出的维度为(num_samples, pred_len)
            predictions = np.empty((X.shape[0], pred_len))
            for i in range(pred_len):
                # 使用当前窗口进行预测
                current_window = X[:, -self.seq_len:]
                pred = self.model.predict(current_window)
                # 更新预测结果
                predictions[:, i] = pred.reshape(-1)
                # 更新输入数据以包含最新预测
                X = np.concatenate((X, pred.reshape(-1, 1)), axis=1)
            return predictions

        elif self.msas=="MIMO":# MIMO预测
            assert pred_len == self.pred_len, "预测长度与模型训练时的长度不一致"
            # 使用模型直接进行预测
            current_window = X[:, -self.seq_len:]
            predictions = self.model.predict(current_window)
            return predictions




class DoubleExponentialSmoothing(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.alpha = args.alpha  # 水平平滑参数
        self.beta = args.beta    # 趋势平滑参数
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.level = None
        self.trend = None
        self.msas= args.msas


    def _fit(self, X: np.ndarray) -> None:
        X_target = X[:, :, -1]

        # 初始化水平和趋势数组
        self.level = np.zeros(X_target.shape)
        self.trend = np.zeros(X_target.shape)
        self.level[:, 0] = X_target[:, 0]
        self.trend[:, 0] = X_target[:, 1] - X_target[:, 0]

        # 计算水平和趋势
        for i in range(1, X_target.shape[1]):
            self.level[:, i] = self.alpha * X_target[:, i] + (1 - self.alpha) * (self.level[:, i-1] + self.trend[:, i-1])
            self.trend[:, i] = self.beta * (self.level[:, i] - self.level[:, i-1]) + (1 - self.beta) * self.trend[:, i-1]

        # 使用水平和趋势作为特征进行预测
        features = np.concatenate([self.level, self.trend], axis=1)
        if self.msas=="recursive":
            subseries = [sliding_window_view(v, self.seq_len + 1) for v in features]
        elif self.msas == "MIMO":
            subseries = [sliding_window_view(v, self.seq_len + self.pred_len) for v in features]
        subseries = np.concatenate(subseries, axis=0)
        train_X = subseries[:, :self.seq_len]
        train_Y = subseries[:, self.seq_len:]
        self.model = linear_model.LinearRegression()
        self.model.fit(train_X, train_Y)



    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        if self.msas=="recursive":
            predictions = np.empty((X.shape[0], pred_len))
            for i in range(pred_len):
                # 使用当前窗口进行预测
                current_window = X[:, -self.seq_len:]
                pred = self.model.predict(current_window)
                # 更新预测结果
                predictions[:, i] = pred.reshape(-1)
                # 更新输入数据以包含最新预测
                X = np.concatenate((X, pred.reshape(-1, 1)), axis=1)
            return predictions
        elif self.msas == "MIMO":
            assert pred_len == self.pred_len, "预测长度与模型训练时的长度不一致"
            # 使用模型直接进行预测
            current_window = X[:, -self.seq_len:]
            predictions = self.model.predict(current_window)
            return predictions










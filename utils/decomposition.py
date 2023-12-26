
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from statsmodels.tsa.seasonal import STL,seasonal_decompose

def get_decompose(args):
    decompose_dict = {
        'STL': STL_decomposition,
        'MA': moving_average,
        'MA_r':moving_average_with_resid,
        'Diff': differential_decomposition,
        'None': None,
    }
    return decompose_dict[args.decompose]


# 移动平均分解
def moving_average(x, seasonal_period):
    """
    Moving Average Algorithm
    Args:
        x (numpy.ndarray): Input time series data, shape (num_samples, num_channels)
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component, shape (num_samples, num_channels)
        seasonal (numpy.ndarray): Seasonal component, shape (num_samples, num_channels)
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    """
    兼容seasonal_period的奇偶性
    """
    x_padded = F.pad(x_tensor.unsqueeze(0), ((seasonal_period - 1) // 2, seasonal_period // 2), mode='reflect').squeeze(0)  # 应用反射填充
    # x_padded = F.pad(x_tensor.unsqueeze(0), ((seasonal_period - 1) // 2, seasonal_period // 2), mode='constant', value=0).squeeze(0)# 0填充
    model = nn.AvgPool1d(kernel_size=seasonal_period, stride=1, padding=0)
    moving_avg = model(x_padded.unsqueeze(0)).squeeze(0)#unsqueeze(0)在第0维增加一个维度，squeeze(0)去掉第0维
    trend = moving_avg.numpy()

    #不算残差，即残差为0，都加到季节性中
    seasonal = (x_tensor - moving_avg).numpy()

    residual = x - trend - seasonal
    return trend, seasonal,residual

def moving_average_with_resid(x, seasonal_period):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    x_padded = F.pad(x_tensor.unsqueeze(0), ((seasonal_period - 1) // 2, seasonal_period // 2), mode='reflect').squeeze(0)  # 应用反射填充
    model = nn.AvgPool1d(kernel_size=seasonal_period, stride=1, padding=0)
    moving_avg = model(x_padded.unsqueeze(0)).squeeze(0)#unsqueeze(0)在第0维增加一个维度，squeeze(0)去掉第0维
    trend = moving_avg.numpy()

    # 计算残差分量
    seasonal_estimate = x - trend
    seasonal = np.array([np.mean(seasonal_estimate[i::seasonal_period]) for i in range(seasonal_period)] * (
                len(x) // seasonal_period + 1))[:len(x)]# 调整季节分量使其周期性

    residual = x - trend - seasonal
    return trend, seasonal,residual

#通过差分时间序列数据来分离趋势和季节性成分
def differential_decomposition(x, seasonal_period):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data, shape (num_samples, num_channels)
    Returns:
        trend (numpy.ndarray): Trend component, shape (num_samples, num_channels)
        seasonal (numpy.ndarray): Seasonal component, shape (num_samples, num_channels)
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)

    #法一
    # x_diff = x_tensor[ 1:] - x_tensor[ :-1]# 一阶差分去除趋势
    # trend = F.pad(x_diff, (1, 0),mode='constant', value=0).numpy()

    #法二
    x_diff = x_tensor[ 1:] - x_tensor[:-1]
    seasonal_diff = x_diff[seasonal_period:] - x_diff[:-seasonal_period]# 季节性差分去除季节性
    trend = x_tensor - torch.cat((torch.zeros(seasonal_period+1), seasonal_diff)).cumsum(dim=0).numpy()


    seasonal = x_tensor.numpy() - trend.numpy()
    return trend.numpy(), seasonal


def STL_decomposition(x, seasonal_period):
    trends = np.zeros_like(x)
    seasonals = np.zeros_like(x)
    # residuals = np.zeros_like(x)

    # for i in range(x.shape[-1]):
    #     result = STL(x[:, i], period=seasonal_period).fit
    #     trends[:, i] = result.trend
    #     seasonals[:, i] = result.seasonal
    #     # residuals[:, i] = result.resid

    #单变量
    result = STL(x, period=seasonal_period,seasonal=15,robust=True).fit()#trend=49,
    trends = result.trend
    seasonals = result.seasonal
    residuals = result.resid

    return trends, seasonals,residuals



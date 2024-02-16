import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.seasonal import STL,seasonal_decompose
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
def get_decompose(args):
    decompose_dict = {
        'STL': STL_decomposition,
        'STL_r': STL_robust,
        'STL_s': STL_seasonal,
        'STL_s_r': STL_seasonal_robust,
        # 'MA': moving_average,
        # 'MA_r':moving_average_with_resid,
        'MA': moving_average_series,
        'Diff': differential_decomposition,
        'X11': X11_decomposition,
        'None': None,
    }
    return decompose_dict[args.decompose]
#一个时间装饰器
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {(end - start):.2f} seconds')
        return result
    return wrapper

# 移动平均分解
# def moving_average(x, seasonal_period):
#     """
#     Moving Average Algorithm
#     Args:
#         x (numpy.ndarray): Input time series data, shape (num_samples, num_channels)
#         seasonal_period (int): Seasonal period
#     Returns:
#         trend (numpy.ndarray): Trend component, shape (num_samples, num_channels)
#         seasonal (numpy.ndarray): Seasonal component, shape (num_samples, num_channels)
#     """
#     x_tensor = torch.tensor(x, dtype=torch.float32)
#     """
#     兼容seasonal_period的奇偶性
#     """
#     x_padded = F.pad(x_tensor.unsqueeze(0), ((seasonal_period - 1) // 2, seasonal_period // 2), mode='reflect').squeeze(0)  # 应用反射填充
#     # x_padded = F.pad(x_tensor.unsqueeze(0), ((seasonal_period - 1) // 2, seasonal_period // 2), mode='constant', value=0).squeeze(0)# 0填充
#     model = nn.AvgPool1d(kernel_size=seasonal_period, stride=1, padding=0)
#     moving_avg = model(x_padded.unsqueeze(0)).squeeze(0)#unsqueeze(0)在第0维增加一个维度，squeeze(0)去掉第0维
#     trend = moving_avg.numpy()
#
#     #不算残差，即残差为0，都加到季节性中
#     seasonal = (x_tensor - moving_avg).numpy()
#
#     residual = x - trend - seasonal
#     return trend, seasonal,residual
#
# def moving_average_with_resid(x, seasonal_period):
#     x_tensor = torch.tensor(x, dtype=torch.float32)
#     x_padded = F.pad(x_tensor.unsqueeze(0), ((seasonal_period - 1) // 2, seasonal_period // 2), mode='reflect').squeeze(0)  # 应用反射填充
#     model = nn.AvgPool1d(kernel_size=seasonal_period, stride=1, padding=0)
#     moving_avg = model(x_padded.unsqueeze(0)).squeeze(0)#unsqueeze(0)在第0维增加一个维度，squeeze(0)去掉第0维
#     trend = moving_avg.numpy()
#
#     # 计算残差分量
#     seasonal_estimate = x - trend
#     seasonal = np.array([np.mean(seasonal_estimate[i::seasonal_period]) for i in range(seasonal_period)] * (
#                 len(x) // seasonal_period + 1))[:len(x)]# 调整季节分量使其周期性
#
#     residual = x - trend - seasonal
#     return trend, seasonal,residual
# @timeit
def moving_average_series(x,seasonal_period,resid):
    """
    Moving average block to highlight the trend of time series
    """
    # x 的形状为 (batch_size, seq_len, channels)
    #检查x是否是tensor，如果不是，转为tensor
    if not isinstance(x,torch.Tensor):
        x=torch.tensor(x, dtype=torch.float32)

    # Initialize components
    trend = torch.zeros_like(x)
    seasonal = torch.zeros_like(x)
    residual = torch.zeros_like(x)

    if len(x.shape)==2:
    # Calculate components for each channel
        for channel in range(x.shape[-1]):
            x_channel = x[:, channel]

            # Calculate moving average
            window_size = seasonal_period
            padding = window_size // 2
            x_padded = F.pad(x_channel.unsqueeze(0).unsqueeze(0), (padding-1, padding), mode='reflect').squeeze(0)
            moving_avg = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1, padding=0).squeeze()

            # Adjust trend size to match original data size
            trend_size = moving_avg.shape[0]
            trend[:, channel] = moving_avg

            # Deseasonalize the series by removing the trend component
            deseasonalized = x_channel - trend[:trend_size, channel]

            # Calculate seasonal component by averaging over the same seasonal periods
            for i in range(seasonal_period):
                seasonal_indices = torch.arange(i, trend_size, seasonal_period)
                if len(seasonal_indices) > 0:
                    seasonal_mean = deseasonalized[seasonal_indices].mean()
                    seasonal[seasonal_indices, channel] = seasonal_mean

            # Calculate residual component
            residual[:, channel] = x_channel - trend[:trend_size, channel] - seasonal[:trend_size, channel]
    else:
        batch_size,seq_len,channels = x.shape[0],x.shape[1],x.shape[2]
        for j in range(batch_size):
            for i in range(channels):
                x_channel = x[j, :, i]
                # Calculate moving average
                window_size = seasonal_period
                padding = window_size // 2
                x_padded = F.pad(x_channel.unsqueeze(0).unsqueeze(0), (padding-1, padding), mode='reflect').squeeze(0)
                moving_avg = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1, padding=0).squeeze()

                # Adjust trend size to match original data size
                trend_size = moving_avg.shape[0]
                trend[j, :, i] = moving_avg

                # Deseasonalize the series by removing the trend component
                deseasonalized = x_channel - trend[j, :trend_size, i]

                # Calculate seasonal component by averaging over the same seasonal periods
                for k in range(seasonal_period):
                    seasonal_indices = torch.arange(k, trend_size, seasonal_period)
                    if len(seasonal_indices) > 0:
                        seasonal_mean = deseasonalized[seasonal_indices].mean()
                        seasonal[j, seasonal_indices, i] = seasonal_mean

                # Calculate residual component
                residual[j, :, i] = x_channel - trend[j, :trend_size, i] - seasonal[j, :trend_size, i]

    # Convert components to numpy arrays
    # trend = trend.numpy()
    # seasonal = seasonal.numpy()
    # residual = residual.numpy()

    if resid:
        return trend, seasonal, residual
    else:
        return trend, seasonal+residual, residual

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

# @timeit
def STL_decomposition(x, seasonal_period, resid):
    #如果x是tensor，转为numpy
    if isinstance(x,torch.Tensor):
        x=x.cpu().numpy()
    trends = np.zeros_like(x)
    seasonals = np.zeros_like(x)
    residuals = np.zeros_like(x)

    if len(x.shape)==2:
        for i in range(x.shape[-1]):
            result = STL(x[:, i], period=seasonal_period).fit()
            trends[:, i] = result.trend
            seasonals[:, i] = result.seasonal
            residuals[:, i] = result.resid
    elif len(x.shape)==1:
        result = STL(x, period=seasonal_period).fit()
        trends = result.trend
        seasonals = result.seasonal
        residuals = result.resid
    else:
        batch_size,seq_len,channels = x.shape[0],x.shape[1],x.shape[2]
        for j in range(batch_size):
            for i in range(channels):
                result = STL(x[j, :, i], period=seasonal_period).fit()
                trends[j, :, i] = result.trend
                seasonals[j, :, i] = result.seasonal
                residuals[j, :, i] = result.resid

    if resid:
        return trends, seasonals, residuals
    else:
        return trends, seasonals + residuals, residuals

def STL_robust(x, seasonal_period, resid):
    #如果x是tensor，转为numpy
    if isinstance(x,torch.Tensor):
        x=x.cpu().numpy()

    if x.shape[-1]>1:
        trends = np.zeros_like(x)
        seasonals = np.zeros_like(x)
        residuals = np.zeros_like(x)
        for i in range(x.shape[-1]):
            result = STL(x[:, i], period=seasonal_period, robust=True).fit()
            trends[:, i] = result.trend
            seasonals[:, i] = result.seasonal
            residuals[:, i] = result.resid
    else:
        x = x.reshape(-1)
        # 单变量
        result = STL(x, period=seasonal_period, robust=True).fit()  # trend=49,
        trends = result.trend
        seasonals = result.seasonal
        residuals = result.resid

    if resid:
        return trends, seasonals, residuals
    else:
        return trends, seasonals + residuals, residuals

def STL_seasonal(x, seasonal_period, resid):
    #如果x是tensor，转为numpy
    if isinstance(x,torch.Tensor):
        x=x.cpu().numpy()

    if x.shape[-1]>1:
        trends = np.zeros_like(x)
        seasonals = np.zeros_like(x)
        residuals = np.zeros_like(x)
        for i in range(x.shape[-1]):
            result = STL(x[:, i], period=seasonal_period, seasonal=15).fit()
            trends[:, i] = result.trend
            seasonals[:, i] = result.seasonal
            residuals[:, i] = result.resid
    else:
        x = x.reshape(-1)
        # 单变量
        result = STL(x, period=seasonal_period, seasonal=15).fit()  # trend=49,
        trends = result.trend
        seasonals = result.seasonal
        residuals = result.resid

    if resid:
        return trends, seasonals, residuals
    else:
        return trends, seasonals + residuals, residuals

def STL_seasonal_robust(x, seasonal_period, resid):
    #如果x是tensor，转为numpy
    if isinstance(x,torch.Tensor):
        x=x.cpu().numpy()

    if len(x.shape[-1])==2:
        trends = np.zeros_like(x)
        seasonals = np.zeros_like(x)
        residuals = np.zeros_like(x)
        for i in range(x.shape[-1]):
            result = STL(x[:, i], period=seasonal_period, seasonal=15, robust=True).fit()
            trends[:, i] = result.trend
            seasonals[:, i] = result.seasonal
            residuals[:, i] = result.resid
    else:
        x = x.reshape(-1)
        # 单变量
        result = STL(x, period=seasonal_period, seasonal=15, robust=True).fit()  # trend=49,
        trends = result.trend
        seasonals = result.seasonal
        residuals = result.resid

    if resid:
        return trends, seasonals, residuals
    else:
        return trends, seasonals + residuals, residuals
@timeit
def X11_decomposition(x, seasonal_period, resid):
    # 如果x是tensor，转为numpy
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    trends = np.zeros_like(x)
    seasonals = np.zeros_like(x)
    residuals = np.zeros_like(x)
    if len(x.shape)==2:
        for i in range(x.shape[-1]):
            xi = pd.Series(x[:, i])# 将每个通道的数据转换为pandas序列
            result = seasonal_decompose(xi, model='additive', period=seasonal_period, extrapolate_trend='freq')
            trends[:, i] = result.trend
            seasonals[:, i] = result.seasonal
            residuals[:, i] = result.resid

    else:
        batch_size,seq_len,channels = x.shape[0],x.shape[1],x.shape[2]
        for j in range(batch_size):
            for i in range(channels):
                xi = pd.Series(x[j, :, i])
                result = seasonal_decompose(xi, model='additive', period=seasonal_period, extrapolate_trend='freq')
                trends[j, :, i] = result.trend
                seasonals[j, :, i] = result.seasonal
                residuals[j, :, i] = result.resid
    # 将结果中的NaN值替换为0，因seasonal_decompose在趋势和残差的边界处可能产生NaN值
    trends = np.nan_to_num(trends)
    seasonals = np.nan_to_num(seasonals)
    residuals = np.nan_to_num(residuals)

    if resid:
        return trends, seasonals, residuals
    else:
        return trends, seasonals + residuals, residuals

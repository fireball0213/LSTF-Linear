
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.seasonal import STL,seasonal_decompose
import numpy as np
import pandas as pd
def get_decompose(args):
    decompose_dict = {
        'STL': STL_decomposition,
        'MA': moving_average,
        'MA_r':moving_average_with_resid,
        'MA_s': moving_average_series,
        'Diff': differential_decomposition,
        'X11': X11_decomposition,
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

def moving_average_series(x,seasonal_period):
    """
    Moving average block to highlight the trend of time series
    """
    # x 的形状为 (batch_size, seq_len, channels)
    #检查x是否是tensor，如果不是，转为tensor
    if not isinstance(x,torch.Tensor):
        x=torch.tensor(x, dtype=torch.float32)
    # 初始化趋势和季节性分量的张量
    trend = torch.zeros_like(x)
    seasonal = torch.zeros_like(x)
    #如果x只有两个维度，则展平为一维，x形状变为 (seq_len)后正常分解
    if len(x.shape)==2:
        # x=x.unsqueeze(0)
        x_padded = F.pad(x, (seasonal_period) // 2, mode='reflect')
        moving_avg = F.avg_pool1d(x_padded, kernel_size=seasonal_period, stride=1, padding=0)
        trend = moving_avg.squeeze(1)  # 移除额外的维度，并保存趋势分量
        seasonal_estimate = x.squeeze(1) - trend
        # 对季节性分量进行调整
        for j in range(seasonal_period):
            seasonal[:, j::seasonal_period] = seasonal_estimate[:, j::seasonal_period].mean(dim=1, keepdim=True)

    else:
        for i in range(x.shape[2]):  # 遍历每个通道
            x_channel = x[:, :, i].unsqueeze(1)  # 为每个通道增加一个维度，形状变为 (batch_size, 1, seq_len)
            # 对每个通道的数据应用填充和平均池化
            x_padded = F.pad(x_channel, ((seasonal_period - 1) // 2, seasonal_period // 2), mode='reflect')
            moving_avg = F.avg_pool1d(x_padded, kernel_size=seasonal_period, stride=1, padding=0)

            trend[:, :, i] = moving_avg.squeeze(1)  # 移除额外的维度，并保存趋势分量
            seasonal_estimate = x_channel.squeeze(1) - trend[:, :, i]
            # 对季节性分量进行调整
            for j in range(seasonal_period):
                seasonal[:, j::seasonal_period, i] = seasonal_estimate[:, j::seasonal_period].mean(dim=1, keepdim=True)

    residual = x - trend - seasonal
    return trend, seasonal, residual

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
    #如果x是tensor，转为numpy
    if isinstance(x,torch.Tensor):
        x=x.cpu().numpy()
    x=x.reshape(-1)
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




def X11_decomposition(x, seasonal_period):
    # 如果x是tensor，转为numpy
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    x = x.reshape(-1)

    # 将numpy数组转换为pandas序列，确保时间序列索引
    x = pd.Series(x)

    # 使用seasonal_decompose方法进行季节性分解，模式设为'additive'或'multiplicative'，取决于数据特性
    # 此处使用'additive'作为示例
    result = seasonal_decompose(x, model='additive', period=seasonal_period)

    trends = result.trend
    seasonals = result.seasonal
    residuals = result.resid

    # 将结果中的NaN值替换为0，因seasonal_decompose在趋势和残差的边界处可能产生NaN值
    trends = np.nan_to_num(trends)
    seasonals = np.nan_to_num(seasonals)
    residuals = np.nan_to_num(residuals)

    return trends, seasonals, residuals

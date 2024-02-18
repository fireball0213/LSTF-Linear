
# -*- coding: UTF-8 -*- #
"""
@filename:data_tools.py
@author:201300086
@time:2023-03-13
"""
import numpy as np
import torch
import datetime


def time_features(dates, timeenc=1, freq='h'):
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0:
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    >
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]):
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    if timeenc == 0:  # 如果timeenc为0，根据日期提取特定的时间特征
        dates['month'] = dates.date.apply(lambda row: row.month, 1)  # 提取月份
        dates['day'] = dates.date.apply(lambda row: row.day, 1)  # 提取日期
        dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)  # 提取星期几
        dates['hour'] = dates.date.apply(lambda row: row.hour, 1)  # 提取小时
        dates['minute'] = dates.date.apply(lambda row: row.minute, 1)  # 提取分钟
        dates['minute'] = dates.minute.map(lambda x: x // 15)  # 将分钟映射到0-3表示15分钟的一个区间
        freq_map = {
            'y': [], 'M': ['month'], 'w': ['month'], 'd': ['month', 'day', 'weekday'],
            'b': ['month', 'day', 'weekday'], 'h': ['month', 'day', 'weekday', 'hour'],
            'm': ['month', 'day', 'weekday', 'hour', 'minute'],
        }
        return dates[freq_map[freq.lower()]].values  # 返回对应频率的时间特征值
    # if timeenc==1:
    #     dates = pd.to_datetime(dates.date.values)
    #     return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)


# 定义一个标准化类
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    # 将均值和标准差转换为与数据类型和设备相同的tensor
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    # 与transform相反，将标准化后的数据恢复到原始范围
    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def get_one_hot(index, size):
    '''
    获得一个one-hot的编码
    index:编码值
    size:编码长度
    '''
    one_hot = [0 for _ in range(1, size + 1)]  #np.zeros([size], dtype=int)#
    one_hot[index - 1] = 1
    return one_hot


# 定义一个函数用于获得数据的one-hot特征
def get_one_hot_feature(data_stamp, freq,feature):
    if freq == "h" or 'm':  # 如果频率为小时\频率为15分钟
        if feature=='month_week':
            data_one_hot = np.hstack((data_stamp[:, :1], data_stamp[:, 2:3] + 1))  # 处理数据获取one-hot编码需要的部分
            func_ = np.frompyfunc(get_one_hot, 2, 1)  # 使用numpy的frompyfunc函数创建one-hot转换函数
            data_one_hots = func_(data_one_hot, [12, 7])  # 应用转换函数获取one-hot编码
            one_hot=data_one_hots[:,0]+data_one_hots[:,1]  # 合并one-hot编码结果
            one_hot = np.array(one_hot.tolist())  # 将one-hot编码结果转换为numpy数组
        else:#'week'
            data_one_hot =data_stamp[:, 2:3] + 1
            data_one_hots =[get_one_hot(i[0], 7) for i in data_one_hot]  # 应用转换函数获取one-hot编码
            # data_one_hots = get_one_hot(data_one_hot, 7)  # 应用转换函数获取one-hot编码
            one_hot = data_one_hots
        return one_hot  # 返回one-hot编码数组
    else:
        print("freq is not supported")
        return None

def get_sin_cos_feature(data_stamp, freq,feature):
    if freq == "h" or 'm':  # 如果频率为小时\频率为15分钟
        if feature=='week':
            week_num = data_stamp[:, 2:3] + 1
            week = week_num * (2 * np.pi / 7)
            week_sin = np.sin(week)
            week_cos = np.cos(week)
            return np.hstack((week_sin, week_cos))
        elif feature=='month_week':
            month_num = data_stamp[:, :1]
            week_num = data_stamp[:, 2:3] + 1
            month = month_num * (2 * np.pi / 12)
            week = week_num * (2 * np.pi / 7)
            month_sin = np.sin(month)
            month_cos = np.cos(month)
            week_sin = np.sin(week)
            week_cos = np.cos(week)
            return np.hstack((month_sin, month_cos, week_sin, week_cos))
    else:
        print("freq is not supported")
        return None
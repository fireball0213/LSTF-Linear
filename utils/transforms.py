import numpy as np
import torch
from scipy.stats import boxcox
from scipy.special import inv_boxcox
#引入StandardScaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler
import matplotlib.pyplot as plt


class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data,update=False):
        return data

    def inverse_transform(self, data):
        return data


#实现数据的归一化的类，即压缩到0，1之间
class Normalization(Transform):
    def __init__(self, args):
        self.mean=0.
        self.max=1.
        self.min=0.
        pass

    def transform(self, data,update=False):
        if update:
            self.mean=np.mean(data)
            self.max=data.max()
            self.min=data.min()
        #先检测数据的极差是否为0
        if self.max-self.min==0:
            return data-self.mean
        #将数据归一化到0，1之间
        norm_data=(data-self.min)/(self.max-self.min)
        return norm_data

    def inverse_transform(self, data):
        #如果数据极差为0，返回原均值
        if data.max()-data.min()==0:
            return data*self.mean
        #将数据反归一化到原始范围
        inverse_data=data*(self.max-self.min)+self.min
        return inverse_data

#实现数据标准化的类
# class Standardization(Transform):
#     def __init__(self, args):
#         self.mean=0.
#         self.std=1.
#         pass
#
#     def transform(self, data, update=False):
#         if update:
#             self.mean=data.mean()
#             self.std=data.std()
#         #先检测数据的标准差是否为0
#         if self.std==0:
#             return data-self.mean
#         #将数据标准化到0，1之间
#         norm_data=(data-self.mean)/self.std
#         return norm_data
#
#     def inverse_transform(self, data):
#         #如果数据标准差为0，返回原均值
#         if data.std()==0:
#             return data+self.mean
#         #将数据反标准化到原始范围
#         inverse_data=data*self.std+self.mean
#         return inverse_data
#支持多变量的标准化
class Standardization(Transform):
    def __init__(self, args):
        # 初始化均值和标准差为None，它们将是数组
        self.mean = None
        self.std = None

    def transform(self, data, update=False):
        if update:
            # 计算每个变量的均值和标准差
            self.mean = np.mean(data, axis=1)
            self.std = np.std(data, axis=1)
        # 检查标准差是否为0
        if np.any(self.std == 0):
            # 对于标准差为0的变量，只减去均值
            norm_data = np.where(self.std == 0, data - self.mean, data)
        else:
            # 将数据标准化
            norm_data = (data - self.mean) / self.std
            #print(np.mean(norm_data, axis=1),np.std(norm_data, axis=1))#明明多变量是第三个维度，但统计的是第二个维度
        return norm_data

    def inverse_transform(self, data):
        # 反标准化数据
        if self.std is not None:
            # 对于标准差为0的变量，只加上均值
            inverse_data = np.where(self.std == 0, data + self.mean, data * self.std + self.mean)
        else:
            inverse_data = data
        return inverse_data

#实现数据均值归一化的类
class MeanNormalization(Transform):
    def __init__(self, args):
        self.mean=0.
        self.max=1.
        self.min=0.
        pass

    def transform(self, data, update=False):
        if update:
            self.mean=data.mean()
            self.max=data.max()
            self.min=data.min()
        #先检测数据的极差是否为0
        if self.max-self.min==0:
            return data-self.mean
        #将数据归一化到0，1之间
        norm_data=(data-self.mean)/(self.max-self.min)
        return norm_data

    def inverse_transform(self, data):
        #如果数据极差为0，返回原均值
        if data.max()-data.min()==0:
            return data*self.mean
        #将数据反归一化到原始范围
        inverse_data=data*(self.max-self.min)+self.mean
        return inverse_data

# 实现BoxCox的类，注意需要将数据转换为正数
class BoxCox(Transform):
    def __init__(self, args):
        self.lam = 0.
        self.min_val = 0
        self.original_shape = None

    def transform(self, data, update=False):
        # 保存原始数据形状以便逆变换
        self.original_shape = data.shape
        data_flattened = data.ravel()
        if update:
            # 为保证数据为正，找到并记录最小值
            self.min_val = np.min(data_flattened)
            # 将所有数据偏移使其为正
            data_positive = data_flattened - self.min_val + 1
            # 应用 BoxCox 变换
            transformed_data, self.lam = boxcox(data_positive)
        else:
            # 如果不更新，使用已有的lambda和最小值
            data_positive = data_flattened - self.min_val + 1
            transformed_data = boxcox(data_positive, lmbda=self.lam)
        # 将数据还原为原始形状
        return transformed_data.reshape(self.original_shape)

    def inverse_transform(self, data):
        self.original_shape = data.shape
        data_flattened = data.ravel()
        # 应用 BoxCox 逆变换
        data_original = inv_boxcox(data_flattened, self.lam)
        # 将数据偏移还原
        data_original = data_original + self.min_val - 1
        # 将数据还原为原始形状
        return data_original.reshape(self.original_shape)



#实现数据的标准化，使用sklearn中的StandardScaler
class StandardScaler(Transform):
    def __init__(self, args):
        self.scaler=StandardScaler()
        pass

    def transform(self, data,update=False):
        if update:
            self.scaler.fit(data)
        #先检测数据的极差是否为0
        norm_data=self.scaler.transform(data)
        return norm_data

    def inverse_transform(self, data):
        #如果数据极差为0，返回原均值
        inverse_data=self.scaler.inverse_transform(data)
        return inverse_data

#实现数据的归一化，使用sklearn中的MinMaxScaler
class MinMaxScaler(Transform):
    def __init__(self, args):
        self.scaler=MinMaxScaler(feature_range=(0, 1))
        pass

    def transform(self, data,update=False):
        if update:
            self.scaler.fit(data)
        #先检测数据的极差是否为0
        norm_data=self.scaler.transform(data)
        return norm_data

    def inverse_transform(self, data):
        #如果数据极差为0，返回原均值
        inverse_data=self.scaler.inverse_transform(data)
        return inverse_data


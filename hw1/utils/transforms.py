import numpy as np
import torch
from scipy.stats import boxcox
from scipy.special import inv_boxcox
#引入StandardScaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler






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
class Standardization(Transform):
    def __init__(self, args):
        self.mean=0.
        self.std=1.
        pass

    def transform(self, data, update=False):
        if update:
            self.mean=data.mean()
            self.std=data.std()
        #先检测数据的标准差是否为0
        if self.std==0:
            return data-self.mean
        #将数据标准化到0，1之间
        norm_data=(data-self.mean)/self.std
        return norm_data

    def inverse_transform(self, data):
        #如果数据标准差为0，返回原均值
        if data.std()==0:
            return data+self.mean
        #将数据反标准化到原始范围
        inverse_data=data*self.std+self.mean
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
        pass

    def _ensure_three_dimensions(self, data):
        # Ensure data is three-dimensional
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # Add a new dimension at the start
        return data

    def transform(self, data, update=False):
        data = self._ensure_three_dimensions(data)
        data = data - data.min(axis=1, keepdims=True) + 1
        norm_data = np.empty_like(data)

        if update:
            self.lam = np.zeros(data.shape[-1])  # For storing lambda for each feature

        for i in range(data.shape[-1]):
            if update:
                transformed, self.lam[i] = boxcox(data[0, :, i])
                norm_data[0, :, i] = transformed
            else:
                norm_data[0, :, i] = boxcox(data[0, :, i], self.lam[i])

        return np.squeeze(norm_data)  # Remove any singleton dimensions

    def inverse_transform(self, data):
        data = self._ensure_three_dimensions(data)
        inverse_data = np.empty_like(data)

        for i in range(data.shape[-1]):
            inverse_data[0, :, i] = inv_boxcox(data[0, :, i], self.lam[i])

        inverse_data = inverse_data + data.min(axis=1, keepdims=True) - 1
        return np.squeeze(inverse_data)  # Remove any singleton dimensions


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


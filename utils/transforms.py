import numpy as np
import torch
from scipy.stats import boxcox
from scipy.special import inv_boxcox
#引入StandardScaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from dataset.data_visualizer import plot_fft,plot_fft3
import pywt

def get_transform(args):
    transform_dict = {
        'IdentityTransform': IdentityTransform,
        'Normalization': Normalization,
        'Standardization': Standardization,
        'MeanNormalization':MeanNormalization,
        'BoxCox':BoxCox,
        # 'FourierTransform':FourierTransform,
    }
    return transform_dict[args.transform](args)
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
            self.mean_OT=self.mean[0,-1]
            self.std_OT=self.std[0,-1]

        # 检查标准差是否为0
        if np.any(self.std == 0):
            # 对于标准差为0的变量，只减去均值
            norm_data = np.where(self.std == 0, data - self.mean, data)
        else:
            #检查data的维度
            if len(data.shape) == 2:
                norm_data = (data - self.mean_OT) / self.std_OT
            else:
                norm_data = (data - self.mean) / self.std
            #print(np.mean(norm_data, axis=1),np.std(norm_data, axis=1))#明明多变量是第三个维度，但统计的是第二个维度
        return norm_data

    def inverse_transform(self, data):
        #将数据反标准化到原始范围
        if len(data.shape) == 2:
            inverse_data = data * self.std_OT + self.mean_OT
        else:
            inverse_data = data * self.std + self.mean
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




#实现数据向频率域的转换，使用scipy中的fft，仅支持单变量
class FourierTransform:
    def __init__(self, args):
        self.args = args
        self.cutoff_frequency = args.cutoff_frequency  # Define a cutoff frequency
        self.period = args.period
        self.is_cut=0#是否进行了截断
        self.channels_to_denoise = args.channels_to_denoise

    def transform(self, data, update=False):
        transformed_data = []
        for channel in self.channels_to_denoise:
            channel_data = data[0, :, channel]
            self.freq = np.fft.fftfreq(channel_data.shape[0], d=0.5 / self.period)  # Frequency bins
            fft_values = np.fft.fft(channel_data)
            transformed_data.append(fft_values)

            if update:
                pass
        return transformed_data

    def cut(self,transformed_data):
        if self.cutoff_frequency > 0:
            self.is_cut = 1
            self.cutoff_index = np.where(np.abs(self.freq) > self.cutoff_frequency)[0]
            self.saved_high_freqs = []
            for data in transformed_data:
                saved_high_freqs_channel = np.copy(data[self.cutoff_index])
                self.saved_high_freqs.append(saved_high_freqs_channel)
                data[self.cutoff_index] = 0  # Zero out frequencies beyond the cutoff
        return transformed_data

    def inverse_transform(self, original_data, denoised_data):
        reconstructed_data = np.zeros_like(original_data)

        for i, channel in enumerate(self.channels_to_denoise):
            #返还高频部分
            # if self.is_cut:
            #     denoised_data[i][self.cutoff_index] = self.saved_high_freqs[i]
            channel_inverse = np.fft.ifft(denoised_data[i])
            reconstructed_data[0, :, channel] = channel_inverse.real

        for channel in range(original_data.shape[2]):
            if channel not in self.channels_to_denoise:
                reconstructed_data[0, :, channel] = original_data[0, :, channel]

        return reconstructed_data



def denoise_fft(data,args):
    FFT=FourierTransform(args)
    #清洗频谱，去除高频部分
    new_data=FFT.transform(data)
    new_data=FFT.cut(new_data)
    denoise_data=FFT.inverse_transform(data,new_data)
    plot_fft3(data[0, :, -1], denoise_data[0, :, -1], 400,title=args.cutoff_frequency)
    return denoise_data

def get_denoise(args):
    denoise_dict = {
        'None': None,
        'fft': denoise_fft,
        'wavelet': denoise_wavelet,
    }
    return denoise_dict[args.freq_denoise]


def denoise_wavelet(data, args):
    WT = WaveletTransform(args)
    transformed_data = WT.transform(data)
    denoised_data = WT.cut(transformed_data)
    denoised_output = WT.inverse_transform(data, denoised_data)
    return denoised_output

class WaveletTransform:
    def __init__(self, args):
        self.args = args
        self.cutoff_frequency = args.cutoff_frequency  # Define a cutoff frequency
        self.channels_to_denoise = args.channels_to_denoise
        self.wavelet = args.wavelet  # Type of wavelet to use
        # self.level = args.level  # Level of decomposition
        self.is_cut = 0
        self.saved_coeffs = None

    def transform(self, data):
        transformed_data = []
        for channel in self.channels_to_denoise:
            channel_data = data[0, :, channel]
            coeffs = pywt.wavedec(channel_data, self.wavelet, level=1)#level=1表示只分解一次
            transformed_data.append(coeffs)
        return transformed_data

    def cut(self, transformed_data):
        if self.cutoff_frequency > 0:
            self.is_cut = 1
            self.saved_coeffs = []
            for coeffs in transformed_data:
                # Modify coefficients to apply the cutoff
                modified_coeffs = [coeff if i < self.cutoff_frequency else np.zeros_like(coeff)
                                   for i, coeff in enumerate(coeffs)]
                self.saved_coeffs.append(modified_coeffs)
        return transformed_data

    def inverse_transform(self, original_data, denoised_data):
        reconstructed_data = np.zeros_like(original_data)

        for i, channel in enumerate(self.channels_to_denoise):
            # Reconstruct the signal from the denoised coefficients
            channel_inverse = pywt.waverec(denoised_data[i], self.wavelet)
            reconstructed_data[0, :, channel] = channel_inverse[:original_data.shape[2]]

        for channel in range(original_data.shape[2]):
            if channel not in self.channels_to_denoise:
                reconstructed_data[0, :, channel] = original_data[0, :, channel]

        return reconstructed_data

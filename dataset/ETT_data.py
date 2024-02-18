# -*- coding: UTF-8 -*- #
"""
@filename:ETT_data.py
@author:201300086
@time:2023-03-13
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataset.data_tools import  time_features, get_one_hot, get_one_hot_feature
from sklearn.preprocessing import StandardScaler
from dataset.data_visualizer import plot_decompose,plot_spirit
import warnings
from utils.decomposition import get_decompose
warnings.filterwarnings('ignore')
from models.SPIRIT import SPIRITModel
WINDOW = 24
import matplotlib.pyplot as plt


class Dataset_ETT_hour(Dataset):
    def __init__(self,args, flag='train',inverse=False, timeenc=0):
        """
        :param flag: ['train','val','test']
        :param data_path: ['ETTh1.csv','ETTh2.csv','ETTm1.csv','ETTm1.csv']
        :param target:控制目标列
        :param inverse:数据逆向
        :param timeenc:0
        :param freq:暂用h
        """

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = args.target
        # self.ratio_train = args.ratio_train
        # self.ratio_val = args.ratio_val
        # self.ratio_test = args.ratio_test
        self.flag = flag
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = args.frequency
        self.window = args.period
        self.root_path = args.data_path
        self.decompose = get_decompose(args)
        self.period = args.period
        self.scaler = StandardScaler()
        self.args = args
        self.residual=args.residual
        self.use_date=args.use_date
        self.__read_data__()

    # def __read_data__(self):
    #     df_raw = pd.read_csv(self.root_path)
    #
    #     #用于划分
    #     border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
    #     border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    #     border1 = border1s[self.set_type]
    #     border2 = border2s[self.set_type]
    #
    #     if self.features == 'M' or self.features == 'MS':
    #         cols_data = df_raw.columns[1:]
    #         df_data = df_raw[cols_data]
    #     elif self.features == 'S':
    #         df_data = df_raw[[self.target]]
    #
    #     if self.scale:
    #         if self.flag == 'train':
    #             train_data = df_data[border1s[0]:border2s[0]]
    #             self.scaler.fit(train_data.values)
    #         data = self.scaler.transform(df_data.values)
    #     else:
    #         data = df_data.values
    #
    #     df_stamp = df_raw[['date']][border1:border2]
    #     df_stamp['date'] = pd.to_datetime(df_stamp.date)
    #     self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)#月-日-星期-小时
    #     self.data_one_hot = get_one_hot_feature(self.data_stamp, freq=self.freq)
    #     self.data_x = data[border1:border2]
    #
    #     self.data_y = data[border1:border2]
    #
    #     #数据分解
    #     self.trend, self.seasonal, self.resid = self.decompose(self.data_x , self.window)
    #     plot_decompose(self.data_x, self.trend, self.seasonal, self.resid, 0, 1000,'whole decompose_'+str(self.flag))

    def __read_data__(self):
        df_raw = pd.read_csv(self.root_path)

        if self.target=='Multi':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.channel = self.args.channels
        else :
            df_data = df_raw[[self.target]]
            self.channel = 1

        # 根据频率设置训练、验证、测试数据的数量
        if self.freq == 'h':
            self.num_train = 12 * 30 * 24
            self.num_val = 4 * 30 * 24
            self.num_test = 4 * 30 * 24
        elif self.freq == 'm':
            self.num_train = 12 * 30 * 24 * 4
            self.num_val = 4 * 30 * 24 * 4
            self.num_test = 4 * 30 * 24 * 4
        # 根据flag切分数据
        if self.flag == 'train':
            start, end = 0, self.num_train

        elif self.flag == 'val':
            start, end = self.num_train, self.num_train + self.num_val
        else:  # test
            start, end = self.num_train + self.num_val, self.num_train + self.num_val + self.num_test

        #使用训练集进行归一化
        self.scaler.fit(df_data.values[:self.num_train])
        data = self.scaler.transform(df_data.values)

        self.data_train = data[:self.num_train]
        self.data_val = data[self.num_train:self.num_train + self.num_val]
        self.data_test = data[self.num_train + self.num_val:self.num_train + self.num_val + self.num_test]

        self.data_x = data[start:end]
        self.data_y = self.data_x
        self.data_z = self.data_x

        if self.use_date:
            df_stamp = df_raw[['date']][start:end]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)#月-日-星期-小时
            self.data_one_hot = get_one_hot_feature(self.data_stamp, freq=self.freq)

        if self.args.use_spirit:
            self.spirit = SPIRITModel(self.args)
            self.channel=self.args.rank
            if self.flag == 'train':
                x_transformed = self.spirit.fit_transform(self.data_train)
            else:
                _=self.spirit.fit_transform(self.data_train)
                x_transformed = self.spirit.transform(self.data_x)
            self.data_x = x_transformed
            self.data_y = self.data_x
            # self.data_z = self.data_x#在spirit变化后的数据上评估
            # plot_spirit(self.spirit, self.data_train, self.data_test,self.data_x,self.args.rank, self.flag)

        if self.decompose is not None:
            self.trend, self.seasonal, self.resid = self.decompose(self.data_x, self.period,self.residual)
            # plot_decompose(self.data_x, self.trend, self.seasonal, self.resid, 0, 200, 'whole decompose_' + str(self.flag))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_z= self.data_z[r_begin:r_end]
        seq_x_trend = self.trend[s_begin:s_end].reshape(-1, self.channel)
        seq_x_seasonal = self.seasonal[s_begin:s_end].reshape(-1, self.channel)
        seq_x_resid = self.resid[s_begin:s_end].reshape(-1, self.channel)

        if self.use_date:
            seq_x_mark = self.data_one_hot[s_begin]
            seq_y_mark = self.data_one_hot[r_begin]
            #将seq_x_mark和seq_y_mark的维度从(19)变为(19, self.channel)，复制self.channel次
            seq_x_marks = np.tile(seq_x_mark, (self.channel, 1)).T
            seq_y_marks = np.tile(seq_y_mark, (self.channel, 1)).T
            #将时间特征和数据特征合并，类型为np.ndarray，seq_x_mark的维度为(19)，seq_x的维度为(96, 7)，合并后的维度为(96+19, 7)
            seq_x = np.concatenate((seq_x, seq_x_marks), axis=0)
            seq_y = np.concatenate((seq_y, seq_y_marks), axis=0)
            seq_x_trend = np.concatenate((seq_x_trend, seq_y_marks), axis=0)
            seq_x_seasonal = np.concatenate((seq_x_seasonal, seq_y_marks), axis=0)
            seq_x_resid = np.concatenate((seq_x_resid, seq_y_marks), axis=0)

        # seq_x_mark = self.data_one_hot[s_begin:s_end:self.window].reshape(-1, 19)
        # seq_y_mark = self.data_one_hot[r_begin:r_end:self.window].reshape(-1, 19)
        else:
            seq_x_mark, seq_y_mark=0,0

        return seq_x, seq_y, seq_z,seq_x_mark, seq_y_mark, seq_x_trend, seq_x_seasonal, seq_x_resid

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        inverse_data=np.zeros(data.shape)
        if len(data.shape) == 3:
            for i in range(data.shape[0]):
                inverse_data[i] = self.scaler.inverse_transform(data[i])
        else:
            inverse_data = self.scaler.inverse_transform(data)
        return inverse_data


if __name__ == '__main__':
    Data = Dataset_ETT_hour(root_path='dataset/ETT', timeenc=0, scale=True, inverse=False,  # 固定参数
                            features='S', target='OT', freq='h',  # 这三个参数控制分析哪列/哪些列数据，暂定最后一列'OT'
                            flag='train', data_path='ETTh2.csv', size=[24 * 4 * 4, 0, 24 * 4], window=24)  # 可能需要变的
    seq_x, seq_y, seq_x_mark, seq_y_mark = Data[4000]
    print(seq_x, seq_x.shape)
    print(seq_y, seq_y.shape)
    print(seq_x_mark, seq_x_mark.shape)
    print(seq_y_mark, seq_y_mark.shape)
    print(len(Data))

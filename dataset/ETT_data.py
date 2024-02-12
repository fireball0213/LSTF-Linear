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
from dataset.data_tools import StandardScaler, time_features, get_one_hot, get_one_hot_feature
from dataset.data_visualizer import plot_decompose
import warnings
from utils.decomposition import get_decompose
warnings.filterwarnings('ignore')

WINDOW = 24


class Dataset_ETT_hour(Dataset):
    def __init__(self,args, flag='train', size=None,features='S',
                scale=True, inverse=False, timeenc=0):
        """
        :param flag: ['train','val','test']
        :param size:  [seq_len,  pred_len]
        :param features:控制目标列
        :param data_path: ['ETTh1.csv','ETTh2.csv','ETTm1.csv','ETTm1.csv']
        :param target:控制目标列
        :param scale:归一化，默认True
        :param inverse:数据逆向
        :param timeenc:0
        :param freq:暂用h
        """

        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            # self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            # self.label_len = size[1]
            self.pred_len = size[-1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = args.target
        self.scale = scale
        self.flag = flag
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = args.frequency
        self.window = args.period
        self.root_path = args.data_path
        self.decompose = get_decompose(args)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.root_path)

        #用于划分
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)#月-日-星期-小时
        self.data_one_hot = get_one_hot_feature(self.data_stamp, freq=self.freq)
        self.data_x = data[border1:border2]
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        self.data_y = data[border1:border2]

        #数据分解
        self.trend, self.seasonal, self.resid = self.decompose(self.data_x , self.window)
        plot_decompose(self.data_x, self.trend, self.seasonal, self.resid, 0, 1000,'whole decompose_'+str(self.flag))


    def __getitem__(self, index):
        s_begin = index
        # print(self.seq_len)
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = s_end
        r_end = s_end + self.pred_len

        # seq_x = self.data_x[s_begin:s_end].reshape(-1, self.window)
        seq_x = self.data_x[s_begin:s_end]
        # if self.inverse:
        #     seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        # else:
        #     seq_y = self.data_y[r_begin:r_end]
        # seq_y = self.data_y[r_begin:r_end].reshape(-1, self.window)
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end].reshape(-1, self.window, 4)
        # seq_y_mark = self.data_stamp[r_begin:r_end].reshape(-1, self.window, 4)
        seq_x_mark = self.data_one_hot[s_begin:s_end:self.window].reshape(-1, 19)
        seq_y_mark = self.data_one_hot[r_begin:r_end:self.window].reshape(-1, 19)

        seq_x_trend = self.trend[s_begin:s_end]
        seq_x_seasonal = self.seasonal[s_begin:s_end]
        seq_x_resid = self.resid[s_begin:s_end]
        seq_y_trend = self.trend[r_begin:r_end]
        seq_y_seasonal = self.seasonal[r_begin:r_end]
        seq_y_resid = self.resid[r_begin:r_end]


        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_trend, seq_x_seasonal, seq_x_resid, seq_y_trend, seq_y_seasonal, seq_y_resid

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


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

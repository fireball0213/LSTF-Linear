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
from dataset.data_tools import  time_features, get_one_hot, get_one_hot_feature,get_sin_cos_feature
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
        self.use_feature = args.use_feature
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.root_path)
        # 检查数据列缺失情况，打印有缺失值的列
        # print(df_raw.isnull().sum())
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

        if self.use_date is not None:
            df_stamp = df_raw[['date']][start:end]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)  # 月-日-星期-小时
            if self.use_date=='one_hot':
                self.date_embedding = get_one_hot_feature(self.data_stamp, freq=self.freq,feature=self.use_feature)
            elif self.use_date=='sin_cos':
                self.date_embedding = get_sin_cos_feature(self.data_stamp, freq=self.freq,feature=self.use_feature)

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
            self.y_trend, self.y_seasonal, self.y_resid = self.decompose(self.data_y, self.period,self.residual)
            # plot_decompose(self.data_x, self.trend, self.seasonal, self.resid, 0, 200, 'whole decompose_' + str(self.flag))


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_z= self.data_z[r_begin:r_end]
        seq_x_trend = self.trend[s_begin:s_end]#.reshape(-1, self.channel)
        seq_x_seasonal = self.seasonal[s_begin:s_end]#.reshape(-1, self.channel)
        seq_x_resid = self.resid[s_begin:s_end]#.reshape(-1, self.channel)
        seq_y_trend = self.y_trend[r_begin:r_end]#.reshape(-1, self.channel)
        seq_y_seasonal = self.y_seasonal[r_begin:r_end]#.reshape(-1, self.channel)
        seq_y_resid = self.y_resid[r_begin:r_end]#.reshape(-1, self.channel)


        if self.use_date=='one_hot':
            seq_x_mark = self.date_embedding[s_begin]
            seq_y_mark = self.date_embedding[r_begin]
            #将seq_x_mark和seq_y_mark的维度从(19)变为(19, self.channel)，复制self.channel次
            seq_x_marks = np.tile(seq_x_mark, (self.channel, 1)).T
            seq_y_marks = np.tile(seq_y_mark, (self.channel, 1)).T
            #将时间特征和数据特征合并，类型为np.ndarray，seq_x_mark的维度为(19)，seq_x的维度为(96, 7)，合并后的维度为(96+19, 7)
            seq_x = np.concatenate((seq_x, seq_x_marks), axis=0)
            seq_y = np.concatenate((seq_y, seq_y_marks), axis=0)
            seq_x_trend = np.concatenate((seq_x_trend, seq_x_marks), axis=0)
            seq_x_seasonal = np.concatenate((seq_x_seasonal, seq_x_marks), axis=0)
            seq_x_resid = np.concatenate((seq_x_resid, seq_x_marks), axis=0)
        elif self.use_date == 'sin_cos':
            seq_x_mark = self.date_embedding[s_begin:s_end]
            seq_y_mark = self.date_embedding[r_begin:r_end]
            seq_x=np.concatenate((seq_x, seq_x_mark), axis=1)
            seq_y=np.concatenate((seq_y, seq_y_mark), axis=1)
            seq_x_trend = np.concatenate((seq_x_trend, seq_x_mark), axis=1)
            seq_x_seasonal = np.concatenate((seq_x_seasonal, seq_x_mark), axis=1)
            seq_x_resid = np.concatenate((seq_x_resid, seq_x_mark), axis=1)

        else:
            seq_x_mark, seq_y_mark=0,0

        return seq_x, seq_y, seq_z,seq_x_mark, seq_y_mark, seq_x_trend, seq_x_seasonal, seq_y_trend, seq_y_seasonal

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

def merge_weather():
    ett_data_path = "./dataset/ETT/ETTh1.csv"
    weather_data_path = './dataset/weather/weather.csv'
    ett_data = pd.read_csv(ett_data_path)
    weather_data = pd.read_csv(weather_data_path)

    ett_data['date'] = pd.to_datetime(ett_data['date'])
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    ett_data['date_key'] = ett_data['date'].dt.strftime('%m-%d %H:%M')
    weather_data['date_key'] = weather_data['date'].dt.strftime('%m-%d %H:%M')

    # 设置日期键作为索引，以便进行数据对齐
    ett_data_indexed = ett_data.set_index('date_key')
    weather_data_indexed = weather_data.set_index('date_key')

    # 选择需要合并的天气数据列
    weather_columns_to_merge = ['VPmax (mbar)', 'T (degC)', 'p (mbar)', 'Tpot (K)', 'VPdef (mbar)',
                                'VPact (mbar)', 'Tdew (degC)', 'H2OC (mmol/mol)', 'sh (g/kg)']

    # 合并数据
    ett_weather_merged = ett_data_indexed.join(weather_data_indexed[weather_columns_to_merge], on='date_key',
                                               how='left')
    # 输出合并后的数据到文件
    ett_weather_merged.to_csv('./dataset/ETT/ETTh1_weather.csv',index=False)

    # 显示合并后数据的前几行
    print(ett_weather_merged.reset_index(drop=True).head())

# if __name__ == '__main__':
    # Data = Dataset_ETT_hour(root_path='dataset/ETT', timeenc=0, scale=True, inverse=False,  # 固定参数
    #                         features='S', target='OT', freq='h',  # 这三个参数控制分析哪列/哪些列数据，暂定最后一列'OT'
    #                         flag='train', data_path='ETTh2.csv', size=[24 * 4 * 4, 0, 24 * 4], window=24)  # 可能需要变的
    # seq_x, seq_y, seq_x_mark, seq_y_mark = Data[4000]
    # print(seq_x, seq_x.shape)
    # print(seq_y, seq_y.shape)
    # print(seq_x_mark, seq_x_mark.shape)
    # print(seq_y_mark, seq_y_mark.shape)
    # print(len(Data))

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils.metrics import mse, mae, mape, smape, mase
from utils.transforms import get_denoise,get_transform
from models.Transformer import Transformer
from models.PatchTST import PatchTST
from models.DLinear import Linear_NN, NLinear, DLinear
from dataset.data_visualizer import data_vi,plot_fft2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset.data_tools import StandardScaler
import os
from dataset.ETT_data import Dataset_ETT_hour
class MLTrainer:
    def __init__(self, args,model, dataset):
        self.model = model
        self.transform = get_transform(args)
        self.dataset = dataset
        # self.val_dataset = Dataset_ETT_hour(args, flag='val')
        self.period = args.period
        self.distance_dim = args.distance_dim
        self.freq_denoise = get_denoise(args)
        self.args=args
        self.seq_len = args.seq_len
        self.channels = args.channels
        self.pred_len = args.pred_len
        self.decompose_all = args.decompose_all
        self.target = args.target
        self.use_date = args.use_date
        self.use_weather = args.use_weather

        if isinstance(self.model, torch.nn.Module):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备为第0块GPU
            self.model = self.model.cuda()

    def _get_slide_data(self,data):
        if self.target == 'OT':
            subseries = sliding_window_view(data.ravel(), self.seq_len + self.pred_len)
            test_X = subseries[:, :self.seq_len]
            test_Y = subseries[:, self.seq_len:]
        elif self.target == 'Multi':
            subseries_list = [sliding_window_view(data[:, i], self.seq_len + self.pred_len) for i in
                              range(self.channels)]
            subseries = np.stack(subseries_list, axis=-1)
            test_X = subseries[:, :self.seq_len, :]
            test_Y = subseries[:, self.seq_len:, :]
        return test_X, test_Y

    def _get_batch_output_from_flag(self,model, data, data_trend, data_seasonal, flag):
        if flag == None:
            return model(data, data_trend, data_seasonal)
        # #分开预测，则不考虑已分开的DLinear
        elif flag == 'trend':
            return model(data_trend)
        elif flag == 'seasonal':
            return model(data_seasonal)

    def _get_data_from_flag(self,data, flag=None):
        #用于支持趋势和季节性分开预测
        if isinstance(self.model, (Transformer, PatchTST)):
            if self.args.decompose_based and flag == 'trend':
                train_X = data.train_trend
                val_X =data.val_trend
                test_X= data.test_trend
            elif self.args.decompose_based and flag == 'seasonal':
                train_X = data.train_seasonal
                val_X = data.val_seasonal
                test_X = data.test_seasonal
            else:
                train_X = data.data_train
                val_X = data.data_val
                test_X = data.data_test
            return train_X, val_X, test_X
        else:
            if flag == None:
                return data.data_x
            elif flag == 'trend':
                return data.trend
            elif flag == 'seasonal':
                return data.seasonal

    def train(self,flag=None):
        if isinstance(self.model, (DLinear, NLinear)):
        # if isinstance(self.model, torch.nn.Module):#已归一化、计算独热编码
            self.model.train()
            train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
            for train_X, target,_, date_x, date_y,data_trend, data_seasonal,target_trend,target_seasonal in train_loader:
                train_X, target = train_X.to(self.device), target.to(self.device)
                data_trend, data_seasonal, target_trend, target_seasonal = data_trend.to(self.device), data_seasonal.to(self.device), target_trend.to(self.device), target_seasonal.to(self.device)
                if flag=='trend':
                    train_X = data_trend
                    target = target_trend
                elif flag=='seasonal':
                    train_X = data_seasonal
                    target = target_seasonal
                self.model.fit(train_X, target, data_trend, data_seasonal)
        #判断模型是否为Transformer，不能直接通过self.model等于字符串判断
        elif isinstance(self.model, (Transformer, PatchTST)):
            train_X, val_X,_ = self._get_data_from_flag(self.dataset, flag)
            self.model.fit(train_X, val_X, flag)
        else:
            train_X_trend= self._get_data_from_flag(self.dataset, 'trend')
            train_X_seasonal= self._get_data_from_flag(self.dataset, 'seasonal')
            train_X= self._get_data_from_flag(self.dataset,flag)
            # if self.freq_denoise is not None:#对训练数据的选定列去噪
            #     train_X=self.freq_denoise(train_X, self.args)
            # train_X= self.transform.transform(train_X,update=True)
            # data_vi(t_X, 200)
            # plot_fft2(train_X[0, :, -1], self.period,400)
            self.model.fit(train_X,train_X_trend,train_X_seasonal)

    def evaluate(self, test_data,flag=None):
        if isinstance(self.model, (DLinear, NLinear)):
        # if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            fore = []
            test_Y = []
            test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False)
            with torch.no_grad():  # 在评估阶段，不计算梯度
                for data,  target,target_true, date_x, date_y,data_trend, data_seasonal,_,_l in test_loader:
                    data, target_true = data.to(self.device), target_true.to(self.device)
                    if self.use_weather:
                        target_true=target_true[:, :, :self.channels]
                    test_Y.append(target_true.cpu())

                    data_trend, data_seasonal = data_trend.to(self.device), data_seasonal.to(self.device),
                    output = self._get_batch_output_from_flag(self.model, data, data_trend, data_seasonal, flag)
                    if self.use_date=='one_hot':
                        output = output[:, :self.seq_len, :]
                    elif self.use_date=='sin_cos' or self.use_weather:
                        output = output[:, :, :self.channels]
                    fore.append(output.cpu())

            # 聚合所有批次的预测结果和真实标签
            fore = torch.cat(fore, dim=0).numpy()
            test_Y = torch.cat(test_Y, dim=0).numpy()

            #应用逆变换
            # fore = self.dataset.inverse_transform(fore)
            # test_Y = self.dataset.inverse_transform(test_Y)
            if self.args.use_spirit:
                fore = self.dataset.spirit.inverse_transform(fore)
                # test_Y = self.dataset.spirit.inverse_transform(test_Y)


        elif isinstance(self.model, (Transformer, PatchTST)):
            _,_, test_X = self._get_data_from_flag(test_data, flag)
            test_X, _ = self._get_slide_data(test_X)
            _, test_Y = self._get_slide_data(test_data.data_x)#test_Y不分解
            fore = self.model.forecast(test_X)

        else:
            test_data_original= self._get_data_from_flag(test_data)
            test_data_trend= self._get_data_from_flag(test_data, 'trend')
            test_data_seasonal= self._get_data_from_flag(test_data, 'seasonal')
            test_data = self._get_data_from_flag(test_data, flag)

            _,test_Y=self._get_slide_data(test_data_original)
            test_X,_=self._get_slide_data(test_data)
            test_X_trend, _= self._get_slide_data(test_data_trend)
            test_X_seasonal, _ = self._get_slide_data(test_data_seasonal)

            # test_X = self.transform.transform(test_X)
            fore = self.model.forecast(test_X,test_X_trend,test_X_seasonal)
            # fore = self.transform.inverse_transform(fore)


        return fore, test_Y

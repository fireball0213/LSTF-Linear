import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils.metrics import mse, mae, mape, smape, mase
from utils.transforms import get_denoise,get_transform
from dataset.data_visualizer import data_vi,plot_fft2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset.data_tools import StandardScaler
import os
def get_data_from_flag(data,flag):
    if flag == None:
        return data.data_x
    elif flag == 'trend':
        return data.trend
    elif flag == 'seasonal':
        return data.seasonal

def get_batch_output_from_flag(model, data,data_trend, data_seasonal, data_res,flag):
    if flag == None:
        return model(data, data_trend, data_seasonal, data_res)
    # #分开预测，则不考虑已分开的DLinear
    elif flag == 'trend':
        return model(data_trend)
    elif flag == 'seasonal':
        return model(data_seasonal)

class MLTrainer:
    def __init__(self, args,model, dataset,device='cpu'):
        self.model = model
        self.transform = get_transform(args)
        self.dataset = dataset
        self.period = args.period
        self.distance_dim = args.distance_dim
        self.freq_denoise = get_denoise(args)
        self.args=args
        self.seq_len = args.seq_len
        self.channels = args.channels
        self.pred_len = args.pred_len
        self.decompose_all = args.decompose_all
        self.target = args.target

        if isinstance(self.model, torch.nn.Module):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备为第0块GPU
            self.model = self.model.cuda()


    def train(self,flag=None):
        if isinstance(self.model, torch.nn.Module):#已归一化、计算独热编码
            self.model.train()
            train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
            for data, target,_, _, _,data_trend, data_seasonal,data_res in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data_trend, data_seasonal,data_res = data_trend.to(self.device), data_seasonal.to(self.device),data_res.to(self.device)
                self.model.fit(data, target, data_trend, data_seasonal,data_res)
        else:
            train_X= get_data_from_flag(self.dataset,flag)
            if self.freq_denoise is not None:#对训练数据的选定列去噪
                train_X=self.freq_denoise(train_X, self.args)
            t_X = self.transform.transform(train_X,update=True)
            # data_vi(t_X, 200)
            # plot_fft2(train_X[0, :, -1], self.period,400)
            self.model.fit(t_X)

    def evaluate(self, test_data,flag=None):
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            fore = []
            test_Y = []
            test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False)
            with torch.no_grad():  # 在评估阶段，不计算梯度
                for data, target,target_true, _, _,data_trend, data_seasonal,data_res  in test_loader:
                    data, target_true = data.to(self.device), target_true.to(self.device)
                    test_Y.append(target_true.cpu())

                    data_trend, data_seasonal, data_res = data_trend.to(self.device), data_seasonal.to(self.device), data_res.to(self.device)
                    output = get_batch_output_from_flag(self.model, data, data_trend, data_seasonal, data_res, flag)
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
        else:
            test_data = get_data_from_flag(test_data, flag)
            if self.target=='OT':
                subseries=sliding_window_view(test_data.ravel(), self.seq_len + self.pred_len)
                test_X = subseries[:, :self.seq_len]
                test_Y = subseries[:, self.seq_len:]
            elif self.target=='Multi':
                subseries_list = [sliding_window_view(test_data[:, i], self.seq_len + self.pred_len) for i in range(self.channels)]
                subseries = np.stack(subseries_list, axis=-1)
                test_X = subseries[:, :self.seq_len, :]
                test_Y = subseries[:, self.seq_len:, :]
            test_X = self.transform.transform(test_X)
            fore = self.model.forecast(test_X)
            # test_Y = self.transform.transform(test_Y)
            fore = self.transform.inverse_transform(fore)


        return fore, test_Y

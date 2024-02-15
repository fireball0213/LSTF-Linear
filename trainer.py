import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils.metrics import mse, mae, mape, smape, mase
from utils.transforms import get_denoise
from dataset.data_visualizer import data_vi,plot_fft2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset.data_tools import StandardScaler
import os
class MLTrainer:
    def __init__(self, args,model, transform, dataset,device='cpu'):
        self.model = model
        self.transform = transform
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
            # self.args.use_nn_loader = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备为第0块GPU
            self.model = self.model.cuda()


    def train(self,flag=None):
        if isinstance(self.model, torch.nn.Module):#已归一化、计算独热编码
            self.model.train()
            train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
            for data, target, _, _,data_trend, data_seasonal,data_res,target_trend, target_seasonal,_ in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data_trend, data_seasonal,data_res = data_trend.to(self.device), data_seasonal.to(self.device),data_res.to(self.device)
                target_trend, target_seasonal = target_trend.to(self.device), target_seasonal.to(self.device)
                self.model.fit(data, target, data_trend, data_seasonal,data_res, target_trend, target_seasonal)
        else:
            # train_X = self.dataset.train_data
            if flag==None:
                train_X = self.dataset.data_x
            elif flag=='trend':
                train_X = self.dataset.trend
            elif flag=='seasonal':
                train_X = self.dataset.seasonal
            if self.freq_denoise is not None:#对训练数据的选定列去噪
                train_X=self.freq_denoise(train_X, self.args)
            t_X = self.transform.transform(train_X,update=True)
            # data_vi(t_X, 200)
            # plot_fft2(train_X[0, :, -1], self.period,400)
            self.model.fit(t_X)


    # def test(self, dataset, seq_len=96, pred_len=32):
    #     if dataset.type == 'm4':
    #         test_X = dataset.train_data
    #         pred_len = dataset.test_data.shape[-1]
    #     else:
    #         subseries = np.concatenate(([sliding_window_view(v, seq_len + pred_len) for v in dataset.test_data]))
    #         test_X = subseries[:, :seq_len]
    #     fore = self.model.forecast(test_X, pred_len=pred_len)
    #     fore = self.transform.inverse_transform(fore)
    #     return fore

    def evaluate(self, test_data,flag=None):
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            fore = []
            test_Y = []
            test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False)
            with torch.no_grad():  # 在评估阶段，不计算梯度
                for data, target, _, _,data_trend, data_seasonal,data_res,target_trend, target_seasonal,_  in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    # target=target[:, :, -1]
                    data_trend, data_seasonal, data_res = data_trend.to(self.device), data_seasonal.to(self.device), data_res.to(self.device)
                    if flag == None:
                        output = self.model(data,data_trend, data_seasonal, data_res)
                    #分开预测，则不考虑已分开的DLinear
                    elif flag == 'trend':
                        output = self.model(data_trend)
                    elif flag == 'seasonal':
                        output = self.model(data_seasonal)
                    fore.append(output.cpu())
                    test_Y.append(target.cpu())
            # 聚合所有批次的预测结果和真实标签
            fore = torch.cat(fore, dim=0).numpy()
            test_Y = torch.cat(test_Y, dim=0).numpy()
            #应用逆变换
            # fore = self.dataset.inverse_transform(fore)
            # test_Y = self.dataset.inverse_transform(test_Y)
        else:
            if self.target=='OT':
                # test_data = dataset.test_data[:, :, -1]
                # subseries = np.concatenate(([sliding_window_view(v, self.seq_len + self.pred_len) for v in test_data]))
                subseries_Y=sliding_window_view(test_data.data_x.ravel(), self.seq_len + self.pred_len)
                test_Y=subseries_Y[:, self.seq_len:]

                if flag == None:
                    test_data = test_data.data_x
                elif flag == 'trend':
                    test_data = test_data.trend
                elif flag == 'seasonal':
                    test_data = test_data.seasonal
                test_data = test_data.ravel()

                subseries=sliding_window_view(test_data, self.seq_len + self.pred_len)
                test_X = subseries[:, :self.seq_len]

            elif self.target=='Multi':
                if flag == None:
                    test_data = test_data.data_x
                elif flag == 'trend':
                    test_data = test_data.trend
                elif flag == 'seasonal':
                    test_data = test_data.seasonal
                subseries_list = []
                for i in range(self.channels):
                    # 对每个通道的数据应用滑动窗口
                    subseries_list.append(sliding_window_view(test_data[:, i], self.seq_len + self.pred_len))
                subseries = np.stack(subseries_list, axis=-1)

                test_X = subseries[:, :self.seq_len, :]
                test_Y = subseries[:, self.seq_len:, :]
            test_X = self.transform.transform(test_X)
            fore = self.model.forecast(test_X)
            # test_Y = self.transform.transform(test_Y)
            fore = self.transform.inverse_transform(fore)

        return fore, test_Y

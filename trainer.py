import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils.metrics import mse, mae, mape, smape, mase
from utils.transforms import get_denoise
from dataset.data_visualizer import data_vi,plot_fft2
import matplotlib.pyplot as plt
import torch
from dataset.data_tools import StandardScaler
class MLTrainer:
    def __init__(self, args,model, transform, dataset,device):
        self.model = model
        self.transform = transform
        self.dataset = dataset
        self.device = device
        self.period = args.period
        self.distance_dim = args.distance_dim
        self.freq_denoise = get_denoise(args)
        self.args=args
        self.seq_len = args.seq_len
        self.channels = args.channels
        self.pred_len = args.pred_len
        self.decompose_all = args.decompose_all

    def train(self,train_loader):
        if self.args.use_nn_loader:#已归一化、计算独热编码
            self.model.train()
            for data, target, _, _,data_trend, data_seasonal,data_res,target_trend, target_seasonal,_ in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data_trend, data_seasonal,data_res = data_trend.to(self.device), data_seasonal.to(self.device),data_res.to(self.device)
                target_trend, target_seasonal = target_trend.to(self.device), target_seasonal.to(self.device)
                self.model.fit(data, target, data_trend, data_seasonal,data_res, target_trend, target_seasonal)
        else:
            train_X = self.dataset.train_data
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

    def evaluate(self,dataset, test_loader):
        if self.args.use_nn_loader:
            self.model.eval()
            fore = []
            test_Y = []
            with torch.no_grad():  # 在评估阶段，不计算梯度
                for data, target, _, _,data_trend, data_seasonal,data_res,target_trend, target_seasonal,_  in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    data_trend, data_seasonal, data_res = data_trend.to(self.device), data_seasonal.to(self.device), data_res.to(self.device)
                    target_trend, target_seasonal = target_trend.to(self.device), target_seasonal.to(self.device)
                    output = self.model(data,data_trend, data_seasonal, data_res)
                    fore.append(output.cpu())
                    test_Y.append(target.cpu())
            # 聚合所有批次的预测结果和真实标签
            fore = torch.cat(fore, dim=0).numpy()
            test_Y = torch.cat(test_Y, dim=0).numpy()
            #应用逆变换
            fore = self.dataset.inverse_transform(fore)
            test_Y = self.dataset.inverse_transform(test_Y)
        else:
            if dataset.type == 'm4':
                test_X = dataset.train_data
                test_Y = dataset.test_data
                pred_len = dataset.test_data.shape[-1]
            else:
                if self.distance_dim=='OT':
                    test_data = dataset.test_data[:, :, -1]
                    subseries = np.concatenate(([sliding_window_view(v, self.seq_len + self.pred_len) for v in test_data]))
                    test_X = subseries[:, :self.seq_len]
                    test_Y = subseries[:, self.seq_len:]
                elif self.distance_dim=='multi':
                    subseries = np.concatenate(([sliding_window_view(v, (self.seq_len + self.pred_len,1)) for v in dataset.test_data]))
                    subseries = subseries.transpose(0,2,1,3)[:,:,:,0]
                    test_X = subseries[:, :self.seq_len,:]
                    test_Y = subseries[:, self.seq_len:,-1]
            test_X = self.transform.transform(test_X)
            fore = self.model.forecast(test_X, pred_len=self.pred_len)
            # test_Y = self.transform.transform(test_Y)
            fore = self.transform.inverse_transform(fore)

        #保留5位小数
        print('mse:',mse(fore, test_Y).round(5))
        print('mae:', mae(fore, test_Y).round(5))
        print('mape:', mape(fore, test_Y).round(5))
        print('smape:', smape(fore, test_Y).round(5))
        print('mase:', mase(fore, test_Y,season=self.period ).round(5))

        return fore, test_Y

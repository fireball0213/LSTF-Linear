import torch.nn as nn
import numpy as np
import torch
from models.base import MLForecastModel
from numpy.lib.stride_tricks import sliding_window_view
import torch.optim as optim
from utils.decomposition import get_decompose
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.data_visualizer import plot_STL,plot_decompose_batch,plot_decompose
import time

class BaseLinearModel(nn.Module):
    def __init__(self, args):
        super(BaseLinearModel, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.channels
        self.individual = args.individual if hasattr(args, 'individual') else False
        self.use_date = args.use_date
        self.use_feature = args.use_feature
        self.all_channel_loss=args.all_channel_loss
        self.decompose_all = args.decompose_all
        self.use_weather = args.use_weather
        if args.use_spirit:
            self.channels = args.rank

        self.setup_layers()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def setup_layers(self):
        # 定义线性层，具体实现由子类完成
        pass

    def forward(self, x, x_trend=None, x_seasonal=None):
        # 前向传播，具体实现由子类完成
        pass

    def fit(self, x, y,x_trend=None, x_seasonal=None):
        x ,y= x.float(), y.float()
        if self.decompose_all:#使用全部数据的分解结果
            x_trend, x_seasonal= x_trend.float(), x_seasonal.float()
        self.train()
        self.optimizer.zero_grad()
        outputs = self.forward(x,x_trend, x_seasonal)
        loss = self.calculate_loss(outputs, y)
        # print('loss:',loss)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calculate_loss(self, outputs, y):
        if self.individual:
            if self.all_channel_loss:
                return sum([self.criterion(outputs[:, :, i], y[:, :, i]) for i in range(self.final_channels)]) / self.final_channels
            else:
                return sum([self.criterion(outputs[:, :, i], y[:, :, i]) for i in range(self.channels)]) / self.channels
        else:
            if self.all_channel_loss:
                return self.criterion(outputs, y)
            else:
                return self.criterion(outputs[:, :, :self.channels], y[:, :, :self.channels])

    def forecast(self, x, x_trend=None, x_seasonal=None,x_res=None):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions

    def reset_args(self):
        if self.use_date=='one_hot':
            if self.use_feature == 'month_week':
                date_len=19
            elif self.use_feature == 'week':
                date_len=7
            self.final_seq_len = self.seq_len + date_len
            self.final_pred_len = self.pred_len + date_len
            self.final_channels = self.channels
        elif self.use_date=='sin_cos':
            self.final_seq_len = self.seq_len
            self.final_pred_len = self.pred_len
            if self.use_feature=='month_week':
                self.final_channels = self.channels+4
            elif self.use_feature=='week':
                self.final_channels = self.channels+2
            if self.use_weather:
                self.final_channels=self.final_channels+9
        elif self.use_date==None:
            self.final_seq_len = self.seq_len
            self.final_pred_len = self.pred_len
            self.final_channels = self.channels
            if self.use_weather:
                self.final_channels=self.channels+9



class NLinear(BaseLinearModel):
    def setup_layers(self):
        self.reset_args()
        if self.individual:
            self.Linear = nn.ModuleList([nn.Linear(self.final_seq_len, self.final_pred_len) for _ in range(self.final_channels)])
        else:
            # 注意这里调整了output_features的计算方式，以适应你的具体需求
            self.Linear = nn.Linear(self.final_seq_len * self.final_channels, self.final_pred_len * self.final_channels)


    def forward(self, x, x_trend=None, x_seasonal=None):
        x = x.float()
        # 提取最后一个时间步的数据并进行差分操作
        last_index = self.seq_len - 1
        seq_last = x[:, last_index, :].detach()
        seq_last=seq_last.reshape(x.size(0), 1, self.final_channels)
        x = x - seq_last

        if self.individual:
            outputs = torch.zeros(x.size(0), self.final_pred_len, self.final_channels, dtype=x.dtype, device=x.device)
            for i in range(self.final_channels):
                outputs[:, :, i] = self.Linear[i](x[:, :, i])
        else:
            x = x.view(x.size(0),-1)  # Flatten the input
            outputs = self.Linear(x)
            outputs = outputs.view(x.size(0), self.final_pred_len, -1)
        # 将差分操作的影响逆转，恢复到原始数据的相对尺度
        outputs = outputs + seq_last

        return outputs



class DLinear(BaseLinearModel):
    def __init__(self, args):
        super(DLinear, self).__init__(args)
        self.decompose = get_decompose(args)  # 是否考虑趋势和季节性
        self.period = args.period  # 季节性的值
        self.residual = args.residual  # 是否考虑残差
        self.D_N= args.D_N


    def setup_layers(self):
        self.reset_args()
        if self.individual:
            self.Linear_Trend = nn.ModuleList([nn.Linear(self.final_seq_len, self.final_pred_len) for _ in range(self.final_channels)])
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(self.final_seq_len, self.final_pred_len) for _ in range(self.final_channels)])
        else:
            # 为趋势和季节性分量使用单个线性层处理多通道数据
            self.Linear_Trend = nn.Linear(self.final_seq_len * self.final_channels, self.final_pred_len * self.final_channels)
            self.Linear_Seasonal = nn.Linear(self.final_seq_len * self.final_channels , self.final_pred_len * self.final_channels)

    def forward(self, x, x_trend=None, x_seasonal=None):
        if torch.isnan(x_trend).any():
            print("NaN detected in input")

        if self.decompose_all:#使用全部数据的分解结果
            trend, seasonal = x_trend.float(), x_seasonal.float()
        else:#使用局部数据的分解结果
            trend, seasonal, resid = self.decompose(x, self.period,self.residual)
            #如果trend是tensor
            if not isinstance(trend,torch.Tensor):
                trend = torch.tensor(trend, dtype=x.dtype, device=x.device)
                seasonal = torch.tensor(seasonal, dtype=x.dtype, device=x.device)
            trend = trend.float()
            seasonal = seasonal.float()
            # plot_decompose_batch(x, trend, seasonal, resid,  'DLinear')

        if self.individual:
            trend_outputs = torch.zeros(x.size(0), self.final_pred_len, self.final_channels, dtype=x.dtype, device=x.device)
            seasonal_outputs = torch.zeros(x.size(0), self.final_pred_len, self.final_channels, dtype=x.dtype, device=x.device)
            for i in range(self.final_channels):
                trend_channel = trend[:, :, i]
                seasonal_channel = seasonal[:, :, i]
                if self.D_N:
                    if self.use_date is not None and i>=self.channels:
                        if self.use_feature=='month_week' or self.use_feature=='week':
                            trend_outputs[:, :, i] = self.Linear_Trend[i](trend_channel)
                            seasonal_outputs[:, :, i] = self.Linear_Seasonal[i](seasonal_channel)
                    else:
                        last_index = self.seq_len - 1
                        trend_seq_last = trend_channel[:, last_index].reshape(x.size(0), 1).detach()
                        seasonal_seq_last = seasonal_channel[:, last_index].reshape(x.size(0), 1).detach()
                        trend_channel = trend_channel - trend_seq_last
                        seasonal_channel = seasonal_channel - seasonal_seq_last
                        trend_outputs[:, :, i] = self.Linear_Trend[i](trend_channel) + trend_seq_last
                        seasonal_outputs[:, :, i] = self.Linear_Seasonal[i](seasonal_channel) + seasonal_seq_last
                else:
                    trend_outputs[:, :, i] = self.Linear_Trend[i](trend_channel)
                    seasonal_outputs[:, :, i] = self.Linear_Seasonal[i](seasonal_channel)

        else:
            if self.D_N:
                last_index = self.seq_len - 1
                trend_seq_last = trend[:, last_index, :].reshape(x.size(0), 1, self.final_channels).detach()
                seasonal_seq_last = seasonal[:, last_index, :].reshape(x.size(0), 1, self.final_channels).detach()
                trend = trend - trend_seq_last
                seasonal = seasonal - seasonal_seq_last
                trend = trend.view(x.size(0), -1)
                seasonal = seasonal.view(x.size(0), -1)
                trend_outputs = self.Linear_Trend(trend)
                seasonal_outputs = self.Linear_Seasonal(seasonal)
                trend_outputs = trend_outputs.view(x.size(0), self.final_pred_len, self.final_channels)+ trend_seq_last
                seasonal_outputs =seasonal_outputs.view(x.size(0), self.final_pred_len, self.final_channels)+ seasonal_seq_last

            else:
                trend = trend.view(x.size(0), -1)
                seasonal = seasonal.view(x.size(0), -1)
                trend_outputs = self.Linear_Trend(trend).view(x.size(0), self.final_pred_len, self.final_channels)
                seasonal_outputs = self.Linear_Seasonal(seasonal).view(x.size(0), self.final_pred_len, self.final_channels)

        # 合并趋势和季节性分量的预测结果
        outputs = trend_outputs + seasonal_outputs
        return outputs


class Linear_NN(BaseLinearModel):
    def setup_layers(self):
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, x_trend=None, x_seasonal=None,x_res=None):
        x = x.float()
        if self.individual:
            outputs = torch.zeros(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            for i, linear in enumerate(self.Linear):
                outputs[:, :, i] = linear(x[:, :, i])
        else:
            x = x.view(x.size(0), -1)  # Flatten the input
            outputs = self.Linear(x)
            outputs = outputs.view(x.size(0), self.pred_len, 1)
        return outputs




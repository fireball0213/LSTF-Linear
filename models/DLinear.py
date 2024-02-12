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

class BaseLinearModel(nn.Module):
    def __init__(self, args):
        super(BaseLinearModel, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.channels
        self.individual = args.individual if hasattr(args, 'individual') else False
        self.setup_layers()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.decompose_all = args.decompose_all

    def setup_layers(self):
        # 定义线性层，具体实现由子类完成
        pass

    def forward(self, x):
        # 前向传播，具体实现由子类完成
        pass

    def fit(self, x, y,x_trend=None, x_seasonal=None,x_res=None, y_trend=None, y_seasonal=None):
        x ,y= x.float(), y.float()
        if self.decompose_all:
            x_trend, x_seasonal,x_res= x_trend.float(), x_seasonal.float(),x_res.float()
            y_trend, y_seasonal= y_trend.float(), y_seasonal.float()
        self.train()
        self.optimizer.zero_grad()
        if self.decompose_all:#使用全部数据的分解结果
            outputs = self.forward(x,x_trend, x_seasonal,x_res)
        else:#使用局部数据的分解结果
            outputs = self.forward(x)
        loss = self.calculate_loss(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calculate_loss(self, outputs, y):
        if self.individual:
            return sum([self.criterion(outputs[:, :, i], y[:, :, i]) for i in range(self.channels)]) / self.channels
        else:
            return self.criterion(outputs, y)

    def forecast(self, x, pred_len):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions

class DLinear(BaseLinearModel):
    def __init__(self, args):
        super(DLinear, self).__init__(args)
        self.decompose = get_decompose(args)  # 是否考虑趋势和季节性
        self.period = args.period  # 季节性的值

    def setup_layers(self):
        if self.individual:
            self.Linear_Trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            self.Linear_Trend = nn.Linear(self.seq_len , self.pred_len )
            self.Linear_Seasonal = nn.Linear(self.seq_len , self.pred_len )

    def forward(self, x, x_trend=None, x_seasonal=None,x_res=None):
        if self.decompose_all:
            trend, seasonal ,resid= x_trend.float(), x_seasonal.float(),x_res.float()
        else:
            trend, seasonal, resid = self.decompose(x, self.period)
            #如果trend是tensor
            if isinstance(trend,torch.Tensor):
                trend = trend.float()
                seasonal = seasonal.float()
            # plot_decompose_batch(x, trend, seasonal, resid,  'DLinear')

        if self.individual:
            trend_outputs = torch.zeros(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            seasonal_outputs = torch.zeros(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                trend_outputs[:, :, i] = self.Linear_Trend[i](trend[:, :, i])
                seasonal_outputs[:, :, i] = self.Linear_Seasonal[i](seasonal[:, :, i])
        else:
            trend = trend.view(x.size(0), -1)
            seasonal = seasonal.view(x.size(0), -1)
            trend_outputs = self.Linear_Trend(trend).view(x.size(0), self.pred_len, 1)
            seasonal_outputs = self.Linear_Seasonal(seasonal).view(x.size(0), self.pred_len, 1)

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

    def forward(self, x):
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

class NLinear(BaseLinearModel):
    def setup_layers(self):
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = x.float()
        # 提取最后一个时间步的数据并进行差分操作
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            outputs = torch.zeros(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            for i, linear in enumerate(self.Linear):
                outputs[:, :, i] = linear(x[:, :, i])
        else:
            x = x.view(x.size(0), -1)  # Flatten the input
            outputs = self.Linear(x)
            outputs = outputs.view(x.size(0), self.pred_len, 1)
        # 将差分操作的影响逆转，恢复到原始数据的相对尺度
        outputs = outputs + seq_last
        return outputs




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
        self.setup_layers()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.decompose_all = args.decompose_all
        if args.use_spirit:
            self.channels = args.rank

    def setup_layers(self):
        # 定义线性层，具体实现由子类完成
        pass

    def forward(self, x, x_trend=None, x_seasonal=None,x_res=None):
        # 前向传播，具体实现由子类完成
        pass

    def fit(self, x, y,x_trend=None, x_seasonal=None,x_res=None):
        x ,y= x.float(), y.float()
        if self.decompose_all:#使用全部数据的分解结果
            x_trend, x_seasonal,x_res= x_trend.float(), x_seasonal.float(),x_res.float()
            self.train()
            self.optimizer.zero_grad()
            outputs = self.forward(x,x_trend, x_seasonal,x_res)
        else:#使用局部数据的分解结果
            self.train()
            self.optimizer.zero_grad()
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

    def forecast(self, x, x_trend=None, x_seasonal=None,x_res=None):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions

class NLinear(BaseLinearModel):
    def setup_layers(self):
        input_features = self.seq_len * self.channels  # 确保这与你输入数据的展平形状匹配
        output_features = self.pred_len * (1 if not self.individual else self.channels)  # 根据是否独立处理通道来设置

        if self.individual:
            self.Linear = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            # 注意这里调整了output_features的计算方式，以适应你的具体需求
            self.Linear = nn.Linear(input_features, output_features)


    def forward(self, x, x_trend=None, x_seasonal=None,x_res=None):
        x = x.float()
        # 提取最后一个时间步的数据并进行差分操作
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        if self.individual:
            outputs = torch.zeros(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            # for i, linear in enumerate(self.Linear):
            #     outputs[:, :, i] = linear(x[:, :, i])
            for i in range(self.channels):
                outputs[:, :, i] = self.Linear[i](x[:, :, i])
        else:
            x = x.view(x.size(0),-1)  # Flatten the input
            outputs = self.Linear(x)
            outputs = outputs.view(x.size(0), self.pred_len, -1)
        # 将差分操作的影响逆转，恢复到原始数据的相对尺度
        outputs = outputs + seq_last

        # outputs=outputs[:,:,-1]#.view(x.size(0), self.pred_len, 1)
        return outputs



class DLinear(BaseLinearModel):
    def __init__(self, args):
        super(DLinear, self).__init__(args)
        self.decompose = get_decompose(args)  # 是否考虑趋势和季节性
        self.period = args.period  # 季节性的值
        self.residual = args.residual  # 是否考虑残差


    def setup_layers(self):
        input_features = self.seq_len * self.channels  # 调整输入特征数以匹配多通道数据
        output_features = self.pred_len * self.channels  # 输出特征数也需要调整以保持通道信息
        if self.individual:
            self.Linear_Trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            # 为趋势和季节性分量使用单个线性层处理多通道数据
            self.Linear_Trend = nn.Linear(input_features, output_features)
            self.Linear_Seasonal = nn.Linear(input_features, output_features)

    def forward(self, x, x_trend=None, x_seasonal=None,x_res=None):

        if self.decompose_all:#使用全部数据的分解结果
            trend, seasonal ,resid= x_trend.float(), x_seasonal.float(),x_res.float()
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
            trend_outputs = torch.zeros(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            seasonal_outputs = torch.zeros(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                # trend_outputs[:, :, i] = self.Linear_Trend[i](trend[:, :, i])
                # seasonal_outputs[:, :, i] = self.Linear_Seasonal[i](seasonal[:, :, i])
                trend_channel = trend[:, :, i]
                seasonal_channel = seasonal[:, :, i]

                trend_seq_last = trend_channel[:, -1:]
                seasonal_seq_last = seasonal_channel[:, -1:]

                # trend_channel = trend_channel - trend_seq_last
                # seasonal_channel = seasonal_channel - seasonal_seq_last

                trend_outputs[:, :, i] = self.Linear_Trend[i](trend_channel)
                seasonal_outputs[:, :, i] = self.Linear_Seasonal[i](seasonal_channel)

                # trend_outputs[:, :, i] = trend_outputs[:, :, i] + trend_seq_last
                # seasonal_outputs[:, :, i] = seasonal_outputs[:, :, i] + seasonal_seq_last
        else:
            trend_seq_last = trend[:, -1:,:]
            seasonal_seq_last = seasonal[:, -1:,:]

            trend = trend - trend_seq_last
            seasonal = seasonal - seasonal_seq_last

            trend = trend.view(x.size(0), -1)
            seasonal = seasonal.view(x.size(0), -1)

            trend_outputs = self.Linear_Trend(trend).view(x.size(0), self.pred_len, self.channels)
            seasonal_outputs = self.Linear_Seasonal(seasonal).view(x.size(0), self.pred_len, self.channels)

            trend_outputs = trend_outputs + trend_seq_last
            seasonal_outputs = seasonal_outputs + seasonal_seq_last

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




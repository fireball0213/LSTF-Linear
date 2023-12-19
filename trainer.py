import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from utils.metrics import mse, mae, mape, smape, mase
from utils.transforms import get_denoise
from dataset.data_visualizer import data_vi,plot_fft2
import matplotlib.pyplot as plt
class MLTrainer:
    def __init__(self, args,model, transform, dataset):
        self.model = model
        self.transform = transform
        self.dataset = dataset
        self.period = args.period
        self.distance_dim = args.distance_dim
        self.freq_denoise = get_denoise(args)
        self.args=args

    def train(self):
        train_X = self.dataset.train_data
        if self.freq_denoise is not None:#对训练数据的选定列去噪
            train_X=self.freq_denoise(train_X, self.args)
        t_X = self.transform.transform(train_X,update=True)
        # data_vi(t_X, 200)
        # plot_fft2(train_X[0, :, -1], self.period,400)
        self.model.fit(t_X)


    def test(self, dataset, seq_len=96, pred_len=32):
        if dataset.type == 'm4':
            test_X = dataset.train_data
            pred_len = dataset.test_data.shape[-1]
        else:
            subseries = np.concatenate(([sliding_window_view(v, seq_len + pred_len) for v in dataset.test_data]))
            test_X = subseries[:, :seq_len]
        fore = self.model.forecast(test_X, pred_len=pred_len)
        fore = self.transform.inverse_transform(fore)
        return fore

    def evaluate(self,dataset, seq_len=96, pred_len=32):
        if dataset.type == 'm4':
            test_X = dataset.train_data
            test_Y = dataset.test_data
            pred_len = dataset.test_data.shape[-1]
        else:
            if self.distance_dim=='OT':
                test_data = dataset.test_data[:, :, -1]
                subseries = np.concatenate(([sliding_window_view(v, seq_len + pred_len) for v in test_data]))
                test_X = subseries[:, :seq_len]
                test_Y = subseries[:, seq_len:]
            elif self.distance_dim=='multi':
                subseries = np.concatenate(([sliding_window_view(v, (seq_len + pred_len,1)) for v in dataset.test_data]))
                subseries = subseries.transpose(0,2,1,3)[:,:,:,0]
                test_X = subseries[:, :seq_len,:]
                test_Y = subseries[:, seq_len:,-1]
        test_X = self.transform.transform(test_X)
        fore = self.model.forecast(test_X, pred_len=pred_len)
        # test_Y = self.transform.transform(test_Y)
        fore = self.transform.inverse_transform(fore)

        #保留5位小数
        print('mse:',mse(fore, test_Y).round(5))
        print('mae:', mae(fore, test_Y).round(5))
        print('mape:', mape(fore, test_Y).round(5))
        print('smape:', smape(fore, test_Y).round(5))
        print('mase:', mase(fore, test_Y,season=self.period ).round(5))

        return fore, test_Y

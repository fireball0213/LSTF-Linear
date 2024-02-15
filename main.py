from models.TsfKNN import TsfKNN
from models.baselines import ZeroForecast, MeanForecast
from models.baselines import Autoregression,ExponentialMovingAverage,DoubleExponentialSmoothing,LastValueForecast
from models.DLinear import Linear_NN, NLinear, DLinear
from models.ThetaMethod import ThetaMethodForecast
from sklearn.linear_model import LinearRegression
from models.ResidualModel import ResidualModel
from utils.transforms import IdentityTransform, Normalization, Standardization,MeanNormalization,BoxCox,FourierTransform
from torch.utils.data import DataLoader
from trainer import MLTrainer
from dataset.dataset import get_dataset
from dataset.ETT_data import Dataset_ETT_hour
from dataset.data_visualizer import data_visualize,plot_forecast,plot_day_forecast,plot_random_forecast
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import random
import torch
from models.ARIMA import ARIMAForecast
from utils.metrics import mse, mae, mape, smape, mase,evaluate_metrics
import torch.nn as nn
def get_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--train_data_path', type=str, default='./dataset/m4/Daily-train.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/m4/Daily-test.csv')

    parser.add_argument('--data_path', type=str, default='./dataset/ETT/ETTh1.csv')
    # parser.add_argument('--data_path', type=str, default='./dataset/ETT/ETTh2.csv')
    # parser.add_argument('--data_path', type=str, default='./dataset/ETT/ETTm1.csv')
    # parser.add_argument('--data_path', type=str, default='./dataset/illness/national_illness.csv')
    # parser.add_argument('--data_path', type=str, default='./dataset/electricity/electricity.csv')
    # parser.add_argument('--data_path', type=str, default='./dataset/exchange_rate/exchange_rate.csv')
    # parser.add_argument('--data_path', type=str, default='./dataset/traffic/traffic.csv')
    # parser.add_argument('--data_path', type=str, default='./dataset/weather/weather.csv')

    parser.add_argument('--period', type=int, default=24, help='period used in TsfKNN and MASE,ETT:24,illness:52')

    # parser.add_argument('--dataset', type=str, default='M4', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    # parser.add_argument('--dataset', type=str, default='Custom', help='dataset type, options: [M4, ETT, Custom]')

    # parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--target', type=str, default='Multi', help='target feature')
    parser.add_argument('--ratio_train', type=int, default=0.6, help='train dataset length')
    parser.add_argument('--ratio_val', type=int, default=0.2, help='validate dataset length')
    parser.add_argument('--ratio_test', type=int, default=0.2, help='input sequence length')
    parser.add_argument('--frequency', type=str, default='h', help='frequency of time series data, options: [h, m]')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')


    # model define
    # parser.add_argument('--model', type=str, default='Autoregression', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='TsfKNN', help='model name')#, required=True
    parser.add_argument('--model', type=str, default='NLinear', help='model name')
    # parser.add_argument('--model', type=str, default='DLinear', help='model name')

    # Dlinear define
    parser.add_argument('--channels', type=int, default=7,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--individual', type=bool, default=False, help='individual training for each channel')
    # parser.add_argument('--use_nn_loader', type=bool, default=True, help='Use data loader for PyTorch')
    parser.add_argument('--use_nn_loader', type=bool, default=False, help='Use data loader for PyTorch')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    # decompose method
    parser.add_argument('--decompose_all', type=bool, default=True, help='decompose all series，使用完整序列的分解')
    parser.add_argument('--residual', type=bool, default=False, help='分解后是否将残差分量保留，不保留即加到季节性中')
    parser.add_argument('--decompose', type=str, default='STL')

    #ThetaMethod define
    parser.add_argument('--theta', type=float, default=0, help='Theta parameter for the ThetaMethod')

    #ResidualModel define
    parser.add_argument('--trend_model', type=str, default='Autoregression', help='trend model for ResidualModel')
    parser.add_argument('--seasonal_model', type=str, default='Autoregression', help='seasonal model for ResidualModel')
    # parser.add_argument('--seasonal_model', type=str, default='TsfKNN', help='seasonal model for ResidualModel')
    parser.add_argument('--decompose_based', type=bool, default=False, help='使用两种模型分别预测')

    # EMA define
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha used in ExponentialMovingAverage')
    parser.add_argument('--beta', type=float, default=0.1, help='beta used in DoubleExponentialSmoothing')

    # multi-step ahead strategy used in LR and TsfKNN,多步直接预测or单步迭代预测
    parser.add_argument('--msas', type=str, default='MIMO', help=' options: [MIMO, recursive]')
    # parser.add_argument('--msas', type=str, default='recursive', help='options: [MIMO, recursive]')

    #TsfKNN define
    parser.add_argument('--n_neighbors', type=int, default=9, help='number of neighbors used in TsfKNN')
    parser.add_argument('--distance_dim', type=str, default='OT', help='distance_dim used in TsfKNN, options: [OT, multi]')
    # parser.add_argument('--distance_dim', type=str, default='multi', help='distance_dim used in TsfKNN, options: [OT, multi]')
    # parser.add_argument('--distance', type=str, default='euclidean', help='distance used in TsfKNN')
    parser.add_argument('--distance', type=str, default='manhattan', help='distance used in TsfKNN')
    # parser.add_argument('--distance', type=str, default='chebyshev', help='distance used in TsfKNN')
    # parser.add_argument('--distance', type=str, default='mahalanobis', help='distance used in TsfKNN')
    # parser.add_argument('--distance', type=str, default='weighted_euclidean', help='distance used in TsfKNN')
    # parser.add_argument('--weighted', type=list, default=[1,1,1,1,1,1,1], help='weighted used in weighted_euclidean')
    parser.add_argument('--weighted', type=list, default=[0, 0, 2, 0, 1, 0, 4],help='weighted used in weighted_euclidean')


    # trend predict method used in TsfKNN
    '''
    plain: 只用96个点训练线性模型，预测接下来的32个点
    AR: 用全部trend训练AR模型，再用96个点预测接下来的32个点
    t_plus_s: 在STL计算距离时考虑trend，实际效果基本相当于没做STL
    t_s: 将趋势和季节分量用两个KNN匹配，再相加预测
    AR_AR: 将趋势和季节分量用两个AR模型预测，再相加预测
    '''
    parser.add_argument('--trend', type=str, default='AR_AR', help='options: [plain, AR, t_plus_s, t_s,AR_AR]')

    # transform define
    parser.add_argument('--transform', type=str, default='IdentityTransform',
                        options=['IdentityTransform', 'Normalization', 'Standardization', 'MeanNormalization', 'BoxCox'])

    #freq denoise define
    parser.add_argument('--freq_denoise', type=str, default='None', help='freq_denoise method，options: [None, fft, wavelet]')
    # parser.add_argument('--freq_denoise', type=str, default='fft', help='freq_denoise method，options: [None, fft, wavelet]')
    # parser.add_argument('--freq_denoise', type=str, default='wavelet', help='freq_denoise method，options: [None, fft, wavelet]')
    # parser.add_argument('--channels_to_denoise', type=list, default=[0, 1, 2, 3, 4, 5, 6], help='channels_to_denoise used in fft')
    parser.add_argument('--channels_to_denoise', type=list, default=[6],help='channels_to_denoise used in fft')
    parser.add_argument('--cutoff_frequency', type=float, default=12, help='cutoff_frequency used in fft')
    # parser.add_argument('--cutoff_frequency', type=float, default=0, help='cutoff_frequency used in fft')
    parser.add_argument('--wavelet', type=str, default='db1', help='wavelet used in wavelet')

    #spirit
    parser.add_argument('--use_spirit', default=True, help='Whether to use SPIRIT algorithm')
    parser.add_argument('--spirit_r', type=int, default=10, help='Target dimensionality for SPIRIT algorithm')
    parser.add_argument('--spirit_k', type=int, default=1, help='Number of principal components for SPIRIT')
    parser.add_argument('--spirit_alpha', type=float, default=0.98, help='Forgetting factor for SPIRIT')

    args = parser.parse_args()
    return args


def get_model(args,key=None):
    model_dict = {
        'ZeroForecast': ZeroForecast,
        'MeanForecast': MeanForecast,
        'LastValueForecast': LastValueForecast,
        'Autoregression': Autoregression,
        'ExponentialMovingAverage': ExponentialMovingAverage,
        'DoubleExponentialSmoothing': DoubleExponentialSmoothing,
        'TsfKNN': TsfKNN,
        'Linear_NN':Linear_NN,
        'NLinear' : NLinear,
        'DLinear': DLinear,
        'ARIMA': ARIMAForecast,
        'ThetaMethod':ThetaMethodForecast,
        'ResidualModel':ResidualModel,
    }
    if key is not None:
        return model_dict[key](args)
    return model_dict[args.model](args)


def get_transform(args):
    transform_dict = {
        'IdentityTransform': IdentityTransform,
        'Normalization': Normalization,
        'Standardization': Standardization,
        'MeanNormalization':MeanNormalization,
        'BoxCox':BoxCox,
        # 'FourierTransform':FourierTransform,
    }
    return transform_dict[args.transform](args)

def xxx(args):
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    # 固定随机种子
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.target == 'OT':
        args.channels = 1
    else:
        args.channels = 7

    trend_model = get_model(args, args.trend_model)
    seasonal_model = get_model(args, args.seasonal_model)
    model = get_model(args)

    train_dataset = Dataset_ETT_hour(args, flag='train', scale=True, inverse=False, timeenc=0)
    test_dataset = Dataset_ETT_hour(args, flag='test', scale=True, inverse=False, timeenc=0)
    transform = get_transform(args)

    if not args.decompose_based:  # 使用单一模型
        # train model
        trainer = MLTrainer(args, model=model, transform=transform, dataset=train_dataset)
        trainer.train()

        # data_visualize(dataset.data, 5000)
        # plt.show()

        # print('evaluate model')
        fore, test_Y = trainer.evaluate(test_dataset)
    else:  # 使用两个模型分别预测
        trainer_trend = MLTrainer(args, model=trend_model, transform=transform, dataset=train_dataset)
        trainer_trend.train(flag='trend')

        trainer_seasonal = MLTrainer(args, model=seasonal_model, transform=transform, dataset=train_dataset)
        trainer_seasonal.train(flag='seasonal')

        # print('evaluate model')
        fore_trend, test_Y = trainer_trend.evaluate(test_dataset, flag='trend')
        fore_seasonal, test_Y = trainer_seasonal.evaluate(test_dataset, flag='seasonal')
        fore = fore_trend.reshape(-1, args.pred_len, args.channels) + fore_seasonal.reshape(-1, args.pred_len,
                                                                                            args.channels)
        test_Y = test_Y.reshape(-1, args.pred_len, args.channels)

    evaluate_metrics(fore, test_Y, args,flag="多变量评估")

    # plot_forecast(fore, test_Y, 500)  # 看所有预测结果上的第一个数据点预测的效果
    plot_day_forecast(fore, test_Y)  # 看某日24h的预测结果，每个子图是predict_len个数据点
    plot_random_forecast(fore, test_Y)  # 看随机预测结果上的predict_len个数据点预测的效果
    plt.show()

    evaluate_metrics(fore[:, :, -1], test_Y[:, :, -1], args,flag="OT变量评估")


if __name__ == '__main__':


    args = get_args()
    # args.decompose_based = True
    # args.model = 'ResidualModel'
    args.model = 'DLinear'
    # args.model = 'NLinear'
    # args.model = 'Linear_NN'
    # args.model = 'ThetaMethod'

    # args.trend_model = 'Autoregression'
    # args.seasonal_model = 'Autoregression'
    args.trend_model = 'NLinear'
    args.seasonal_model = 'NLinear'

    args.individual = True
    args.target = 'Multi'
    # args.target = 'OT'

    for args.model,args.decompose_based in [('DLinear',False),('NLinear',True)]:
        for args.individual in [True]:#, False
            print()
            print(args.model," individual =",args.individual)
            for args.decompose in ['STL','MA','X11']:
                for args.residual in [True,False]:
                    print(args.decompose,' resid =',args.residual)
                    xxx(args)

from models.TsfKNN import TsfKNN
from models.baselines import ZeroForecast, MeanForecast
from models.baselines import Autoregression,ExponentialMovingAverage,DoubleExponentialSmoothing,LastValueForecast
from models.DLinear import Linear_NN, NLinear, DLinear
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
    # parser.add_argument('--period', type=int, default=25, help='period used in TsfKNN and MASE,ETT:24,illness:52')
    # parser.add_argument('--period', type=int, default=52, help='period used in TsfKNN and MASE,ETT:24,illness:52')

    # parser.add_argument('--dataset', type=str, default='M4', help='dataset type, options: [M4, ETT, Custom]')
    parser.add_argument('--dataset', type=str, default='ETT', help='dataset type, options: [M4, ETT, Custom]')
    # parser.add_argument('--dataset', type=str, default='Custom', help='dataset type, options: [M4, ETT, Custom]')

    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--ratio_train', type=int, default=0.6, help='train dataset length')
    parser.add_argument('--ratio_val', type=int, default=0, help='validate dataset length')
    parser.add_argument('--ratio_test', type=int, default=0.4, help='input sequence length')
    parser.add_argument('--frequency', type=str, default='h', help='frequency of time series data, options: [h, m]')

    # forcast task config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')


    # model define
    # parser.add_argument('--model', type=str, default='MeanForecast', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='LastValueForecast', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='Autoregression', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='ExponentialMovingAverage', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='DoubleExponentialSmoothing', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='TsfKNN', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='Linear_NN', help='model name')
    # parser.add_argument('--model', type=str, default='NLinear', help='model name')
    parser.add_argument('--model', type=str, default='DLinear', help='model name')

    # Dlinear define
    parser.add_argument('--channels', type=int, default=7,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--individual', type=bool, default=False, help='individual training for each channel')
    parser.add_argument('--use_nn_loader', type=bool, default=True, help='Use data loader for PyTorch')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')


    # decompose method
    parser.add_argument('--decompose_all', type=bool, default=True, help='decompose all series，使用完整序列的分解')
    # parser.add_argument('--decompose_all', type=bool, default=False, help='decompose all series，使用完整序列的分解')
    # parser.add_argument('--decompose', type=str, default='None')
    parser.add_argument('--decompose', type=str, default='STL')
    # parser.add_argument('--decompose', type=str, default='MA', help='不算残差，即残差为0，都加到季节性中')
    # parser.add_argument('--decompose', type=str, default='MA_r', help='带残差的MA')
    # parser.add_argument('--decompose', type=str, default='MA_s', help='适用于seq_len的MA')
    # parser.add_argument('--decompose', type=str, default='Diff', help='差分')
    # parser.add_argument('--decompose', type=str, default='X11', help='X11分解')

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
    # parser.add_argument('--trend', type=str, default='plain', help='options: [plain, AR, t_plus_s, t_s,AR_AR]')#只用96个点训练线性模型，预测接下来的32个点
    parser.add_argument('--trend', type=str, default='AR', help='options: [plain, AR, t_plus_s, t_s,AR_AR]')#用全部trend训练AR模型，再用96个点预测接下来的32个点
    # parser.add_argument('--trend', type=str, default='t_plus_s', help='options: [plain, AR, t_plus_s, t_s,AR_AR]')#在STL计算距离时考虑trend，实际效果基本相当于没做STL
    # parser.add_argument('--trend', type=str, default='t_s', help='options: [plain, AR, t_plus_s, t_s,AR_AR]')#将趋势和季节分量用两个KNN匹配，再相加预测
    # parser.add_argument('--trend', type=str, default='AR_AR', help='options: [plain, AR, t_plus_s, t_s,AR_AR]')#将趋势和季节分量用两个AR模型预测，再相加预测

    # transform define
    # parser.add_argument('--transform', type=str, default='IdentityTransform')
    # parser.add_argument('--transform', type=str, default='Normalization')
    parser.add_argument('--transform', type=str, default='Standardization')
    # parser.add_argument('--transform', type=str, default='MeanNormalization')
    # parser.add_argument('--transform', type=str, default='BoxCox')

    #freq denoise define
    parser.add_argument('--freq_denoise', type=str, default='None', help='freq_denoise method，options: [None, fft, wavelet]')
    # parser.add_argument('--freq_denoise', type=str, default='fft', help='freq_denoise method，options: [None, fft, wavelet]')
    # parser.add_argument('--freq_denoise', type=str, default='wavelet', help='freq_denoise method，options: [None, fft, wavelet]')
    # parser.add_argument('--channels_to_denoise', type=list, default=[0, 1, 2, 3, 4, 5, 6], help='channels_to_denoise used in fft')
    parser.add_argument('--channels_to_denoise', type=list, default=[6],help='channels_to_denoise used in fft')
    parser.add_argument('--cutoff_frequency', type=float, default=12, help='cutoff_frequency used in fft')
    # parser.add_argument('--cutoff_frequency', type=float, default=0, help='cutoff_frequency used in fft')
    parser.add_argument('--wavelet', type=str, default='db1', help='wavelet used in wavelet')

    args = parser.parse_args()
    return args


def get_model(args):
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
    }
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




if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    #固定随机种子
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    args = get_args()


    if args.use_nn_loader:
        # if args.dataset == 'ETT':
        dataset = Dataset_ETT_hour(args, flag='train', size=[args.seq_len, 0, args.pred_len],
                                   features='S', scale=True, inverse=False, timeenc=0)
    else:
        # 使用原始的训练逻辑
        dataset = get_dataset(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备为第0块GPU
    model = model.cuda()

    # train model
    transform = get_transform(args)
    trainer = MLTrainer(args, model=model, transform=transform, dataset=dataset, device=device)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    trainer.train(train_loader)

    # data_visualize(dataset.data, 5000)
    # plt.show()

    # # evaluate model
    print('evaluate model')
    if args.use_nn_loader:
        test_dataset = Dataset_ETT_hour(args, flag='test', size=[args.seq_len, 0, args.pred_len],
                                   features='S', scale=True, inverse=False, timeenc=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        fore,test_Y = trainer.evaluate(dataset, test_loader)
    else:
        fore,test_Y=trainer.evaluate(dataset)

    # plot_forecast(fore, test_Y, 500)  # 看所有预测结果上的第一个数据点预测的效果
    plot_day_forecast(fore, test_Y)  # 看某日24h的预测结果，每个子图是predict_len个数据点
    plot_random_forecast(fore, test_Y)  # 看随机预测结果上的predict_len个数据点预测的效果
    plt.show()




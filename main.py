from models.TsfKNN import TsfKNN
from models.baselines import ZeroForecast, MeanForecast
from models.baselines import Autoregression,ExponentialMovingAverage,DoubleExponentialSmoothing,LastValueForecast
from utils.transforms import IdentityTransform, Normalization, Standardization,MeanNormalization,BoxCox
from trainer import MLTrainer
from dataset.dataset import get_dataset
from dataset.data_visualizer import data_visualize,plot_forecast,plot_all_forecast
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length')


    # model define
    # parser.add_argument('--model', type=str, default='MeanForecast', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='LastValueForecast', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='Autoregression', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='ExponentialMovingAverage', help='model name')#, required=True
    # parser.add_argument('--model', type=str, default='DoubleExponentialSmoothing', help='model name')#, required=True
    parser.add_argument('--model', type=str, default='TsfKNN', help='model name')#, required=True

    # EMA define
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha used in ExponentialMovingAverage')
    parser.add_argument('--beta', type=float, default=0.1, help='beta used in DoubleExponentialSmoothing')

    # multi-step ahead strategy used in LR and TsfKNN,多步直接预测or单步迭代预测
    # parser.add_argument('--msas', type=str, default='MIMO', help=' options: [MIMO, recursive]')
    parser.add_argument('--msas', type=str, default='recursive', help='options: [MIMO, recursive]')

    #TsfKNN define
    parser.add_argument('--n_neighbors', type=int, default=9, help='number of neighbors used in TsfKNN')
    # parser.add_argument('--distance', type=str, default='euclidean', help='distance used in TsfKNN')
    parser.add_argument('--distance', type=str, default='manhattan', help='distance used in TsfKNN')
    # parser.add_argument('--distance', type=str, default='chebyshev', help='distance used in TsfKNN')
    parser.add_argument('--decompose', type=bool, default=True, help='stl_modified distance used in TsfKNN')
    # parser.add_argument('--decompose', type=bool, default=False, help='stl_modified distance used in TsfKNN')
    # trend predict method distance used in stl_modified
    parser.add_argument('--trend', type=str, default='plain', help='options: [plain, AR, STL]')#只用96个点训练线性模型，预测接下来的32个点
    # parser.add_argument('--trend', type=str, default='AR', help='options: [plain, AR, STL]')#用全部trend训练AR模型，再用96个点预测接下来的32个点
    # parser.add_argument('--trend', type=str, default='STL', help='options: [plain, AR, STL]')#在STL计算距离时考虑trend，如果同时考虑了seasonal和resid，相当于没做STL
    # parser.add_argument('--approximate_knn', type=bool, default=False, help='approximate_knn used in TsfKNN')
    parser.add_argument('--approximate_knn', type=bool, default=True, help='approximate_knn used in TsfKNN')
    parser.add_argument('--hash_size', type=int, default=120, help='hash_num used in LSH')#影响查询速度，越大越快但越不准


    # transform define
    # parser.add_argument('--transform', type=str, default='IdentityTransform')
    parser.add_argument('--transform', type=str, default='Normalization')
    # parser.add_argument('--transform', type=str, default='Standardization')
    # parser.add_argument('--transform', type=str, default='MeanNormalization')
    # parser.add_argument('--transform', type=str, default='BoxCox')


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
    }
    return model_dict[args.model](args)


def get_transform(args):
    transform_dict = {
        'IdentityTransform': IdentityTransform,
        'Normalization': Normalization,
        'Standardization': Standardization,
        'MeanNormalization':MeanNormalization,
        'BoxCox':BoxCox,

    }
    return transform_dict[args.transform](args)


if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    args = get_args()
    # load dataset
    dataset = get_dataset(args)

    # data_visualize(dataset, 500)
    plt.show()

    #create model
    model = get_model(args)
    # data transform
    transform = get_transform(args)
    # create trainer
    trainer = MLTrainer(args,model=model, transform=transform, dataset=dataset)
    # # train model
    trainer.train()
    # # evaluate model
    print('evaluate model')
    fore,test_Y=trainer.evaluate(dataset, seq_len=args.seq_len, pred_len=args.pred_len)
    # 画图对比预测结果
    plot_forecast(fore, test_Y,500)
    plot_all_forecast(fore, test_Y)
    plt.show()

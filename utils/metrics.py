import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


def mae(predict, target):
    return np.mean(np.abs(target - predict))


def mape(predict, target):
    # 先找到所有非零的位置
    non_zero = target != 0
    # 初始化 mape_value 数组
    mape_value = np.zeros_like(target)
    # 只在非零位置上计算 MAPE
    mape_value[non_zero] = np.abs((target[non_zero] - predict[non_zero]) / target[non_zero])
    return np.mean(mape_value)




def smape(predict, target):
    denominator = np.abs(target) + np.abs(predict)
    smape_value = np.where(denominator != 0, 2 * np.abs(target - predict) / denominator, 0)
    return np.mean(smape_value)


# def mase(predict, target):
#     return np.mean(np.abs((target - predict) / (np.mean(np.abs(target[1:] - target[:-1])))))

def naive_forecast(y:np.array, season:int=1):
  "naive forecast: season-ahead step forecast, shift by season step ahead"
  return y[:-season]
def mase(predict, target, season=24):
    return mae(target, predict) / mae(target[season:], naive_forecast(target, season))


def evaluate_metrics(fore, test_Y, args,flag=None):
    print(flag,end=" ")
    # 保留5位小数
    # print('mse:', mse(fore, test_Y).round(5),end=" ")
    print('mae:', mae(fore, test_Y).round(5),end=" ")
    # print('mape:', mape(fore, test_Y).round(5),end=" ")
    print('smape:', smape(fore, test_Y).round(5),end=" ")
    print('mase:', mase(fore, test_Y, season=args.period).round(5),end=" ")
    print()
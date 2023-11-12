import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


# TODO: implement the metrics
def mae(predict, target):
    return np.mean(np.abs(target - predict))


def mape(predict, target):
    # 初始化错误数组为 nan
    errors = np.full_like(target, fill_value=np.nan, dtype=np.float64)

    # 找到非零目标值的索引
    non_zero_indices = np.nonzero(target)

    # 只对非零目标值计算百分比误差
    errors[non_zero_indices] = np.abs((target[non_zero_indices] - predict[non_zero_indices]) / target[non_zero_indices])

    # 计算平均值，忽略 nan
    return np.nanmean(errors)


def smape(predict, target):
    denominator = np.abs(target) + np.abs(predict)
    smape_value = np.where(denominator != 0, 2 * np.abs(target - predict) / denominator, 0)
    return np.mean(smape_value)


def mase(predict, target):
    return np.mean(np.abs((target - predict) / (np.mean(np.abs(target[1:] - target[:-1])))))

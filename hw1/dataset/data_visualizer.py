import numpy as np
import matplotlib.pyplot as plt

def data_visualize(dataset, t):
    """
    Choose t continous time points in data and visualize the chosen points. Note that some datasets have more than one
    channel.
    param:
        dataset: dataset to visualize
        t: the number of timestamps to visualize
    """
    #可视化
    #dataset:数据集
    #t:可视化的时间戳数量
    #一个循环，画一个图，将数据集的每个通道都是图中的一条线，并只画t个连续时间点的数据
    for i in range(dataset.data.shape[2]):
        plt.plot(dataset.data[0, :t, i])
    plt.title(dataset.type)
    plt.legend(dataset.data_cols)

    plt.show()

def plot_forecast(fore, test_Y,t):
    #画图对比预测结果，只画第一个通道的结果
    #fore:预测结果，维度是(163,32)
    #test_Y:真实值，维度是(163,32)
    plt.plot(fore[:, 0][:t], label='forecast')
    plt.plot(test_Y[:, 0][:t], label='true')
    plt.legend()
    plt.show()

#画出STL分解后的几个分量图
def plot_STL(stl,t):
    plt.plot(stl.observed[:t],label='original')
    plt.plot(stl.trend[:t],label='trend')
    plt.plot(stl.seasonal[:t],label='seasonal')
    plt.plot(stl.resid[:t],label='resid')
    plt.title("STL decomposition of data")
    plt.legend()
    plt.show()


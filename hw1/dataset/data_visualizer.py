import numpy as np
import matplotlib.pyplot as plt
import random
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

    # plt.show()

def plot_forecast(fore, test_Y,t):
    #画图对比预测结果，只画第一个通道的结果
    #fore:预测结果，维度是(163,32)
    #test_Y:真实值，维度是(163,32)
    plt.plot(fore[:, 0][:t], label='forecast')
    plt.plot(test_Y[:, 0][:t], label='true')
    plt.legend()
    # plt.show()

def plot_all_forecast(fore, test_Y):

    # lst=random.sample(range(0,fore.shape[0]),25)#在range(0,fore.shape[0])中随机选取9个数
    index=random.sample(range(0,fore.shape[0]),1)[0]
    lst=list(range(index,index+25))
    # print(lst)
    #创建3*3的画布，画出9个子图，每个子图对比lst中的一个预测结果和真实值
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.plot(fore[lst[i],: ], label='forecast')
        ax.plot(test_Y[lst[i],: ], label='true')
        ax.legend()
    plt.plot()
    # plt.show()

#画出STL分解后的几个分量图
def plot_STL(stl,t):
    plt.plot(stl.observed[:t],label='original')
    plt.plot(stl.trend[:t],label='trend')
    plt.plot(stl.seasonal[:t],label='seasonal')
    plt.plot(stl.resid[:t],label='resid')
    plt.plot(stl.trend[:t] + stl.seasonal[:t], label='trend+seasonal')
    plt.title("STL decomposition of data")
    plt.legend()
    plt.show()


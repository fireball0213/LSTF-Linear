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



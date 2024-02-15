import numpy as np
import matplotlib.pyplot as plt
import random
import torch
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

def data_vi(dataset, t):
    # for i in range(dataset.shape[2]):
    #     plt.plot(dataset[0, :t, i],label=i)
    # plt.plot(dataset[0, :t, 0], label=0)
    # plt.plot(dataset[0, :t, 1], label=1)
    plt.plot(dataset[0, :t, 6], label=6)
    plt.plot(dataset[0, :t, 2], label=2)
    # plt.plot(dataset[0, :t, 3], label=3)
    plt.plot(dataset[0, :t, 4], label=4)
    plt.legend()
    plt.show()
def plot_forecast(fore, test_Y,t):
    #画图对比预测结果，只画第一个通道的结果
    #fore:预测结果，维度是(163,32)
    #test_Y:真实值，维度是(163,32)
    plt.plot(fore[:, 0][:t], label='forecast')
    plt.plot(test_Y[:, 0][:t], label='true')
    plt.legend()
    # plt.show()
def plot_slide(x,t):
    plt.plot(x[:, 0][:t],label='original')
    plt.legend()
    plt.show()
def plot_day_forecast(fore, test_Y):
    '''
    创建Plot_length*Plot_length的画布,画出Plot_num个子图
    每个子图对比一段长为pred_len的预测结果和真实值，子图间相隔Step个时间点
    '''
    if len(fore.shape)==3:
        fore=fore[:,:,-1]
        test_Y=test_Y[:,:,-1]
    Plot_length=6
    Plot_width=4
    Plot_num=Plot_length*Plot_width
    Step=4
    index=random.sample(range(0,fore.shape[0]-Plot_num*Step),1)[0]
    lst=list(range(index,index+Plot_num*Step,Step))
    fig, axes = plt.subplots(Plot_width, Plot_length, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.plot(fore[lst[i],: ], label='forecast')
        ax.plot(test_Y[lst[i],: ], label='true')
        ax.set_title(str(lst[i]))
        if i == 0:  # 只在第一个子图上添加图例
            ax.legend()
    #使图像title显示在最上面一行

    # plt.title("forecast results"+str(index)+"-"+str(index+Plot_num*Step))
    plt.tight_layout()
    plt.plot()
    # plt.show()

def plot_random_forecast(fore, test_Y):
    '''
    创建Plot_length*Plot_length的画布,画出Plot_num个子图
    每个子图对比一段长为pred_len的预测结果和真实值，子图间相隔Step个时间点
    '''
    if len(fore.shape)==3:
        fore=fore[:,:,-1]
        test_Y=test_Y[:,:,-1]
    Plot_length=6
    Plot_num=Plot_length**2
    lst=random.sample(range(0,fore.shape[0]),Plot_num)
    lst.sort()
    fig, axes = plt.subplots(Plot_length, Plot_length, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.plot(fore[lst[i],: ], label='forecast')
        ax.plot(test_Y[lst[i],: ], label='true')
        ax.set_title(str(lst[i]))
        if i == 0:  # 只在第一个子图上添加图例
            ax.legend()
    # plt.title("forecast results : random view")
    #使图像title不重叠
    plt.tight_layout()
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

def plot_decompose(x,trend,season,resid,t1,t2,model):
    #检查是否是tensor，如果是，转移到cpu，然后转为numpy
    if isinstance(x,torch.Tensor):
        x=x.cpu().numpy()
        trend=trend.cpu().numpy()
        season=season.cpu().numpy()
        resid=resid.cpu().numpy()
    #如果x是三维的，去掉最后一个维度
    if len(x.shape)==3:
        x=x[:,:,0]
    plt.plot(trend[t1:t2,-1],label='trend',color='red')
    plt.plot(season[t1:t2,-1],label='season',color='blue')
    plt.plot(resid[t1:t2,-1],label='resid',color='lightgreen')
    plt.plot(x[t1:t2,-1],label='original',color='grey')
    plt.title(model)
    plt.legend()
    plt.show()

def plot_decompose_batch(x,trend,season,resid,model):
    '''
    x:[batch_size,seq_len,channels]
    '''
    #随机选择4个batch的seq_len个时间点画图
    batch_size=x.shape[0]
    seq_len=x.shape[1]
    channels=x.shape[2]
    #移到cpu
    x=x.cpu().numpy()
    trend=trend.cpu().numpy()
    season=season.cpu().numpy()
    resid=resid.cpu().numpy()
    #画成4个子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        batch=random.randint(0,batch_size-1)
        channel=0
        # t=random.randint(0,seq_len-1)
        ax.plot(trend[batch,:,channel],label='trend',color='red')
        ax.plot(season[batch,:,channel],label='season',color='blue')
        ax.plot(resid[batch,:,channel],label='resid',color='lightgreen')
        ax.plot(x[batch,:,channel],label='original',color='grey')
        ax.set_title(model+'_batch'+str(batch)+'_channel'+str(channel))
        if i == 0:  # 只在第一个子图上添加图例
            ax.legend()
    plt.tight_layout()
    plt.show()
    # plt.plot(trend,label='trend',color='red')
    # plt.plot(season,label='season',color='blue')
    # plt.plot(resid,label='resid',color='lightgreen')
    # plt.plot(x,label='original',color='grey')
    # plt.title(model)
    # plt.legend()
    # plt.show()

def plot_fft(freq,fft_values):
    plt.figure()
    plt.stem(freq[:len(freq) // 2], np.abs(fft_values)[:len(freq) // 2], 'b', markerfmt=" ", basefmt="-b")
    plt.title('Frequency Domain Signal')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()

def plot_fft2(data,period,t):
    '''
    验证频域去噪还原后无损
    '''
    freq = np.fft.fftfreq(data.shape[0], d=0.5 / period)  # Frequency bins
    fft_values = np.fft.fft(data)
    cutoff_index = np.where(np.abs(freq) > 7)[0]
    saved_high_freqs = np.copy(fft_values[cutoff_index])
    fft_values[cutoff_index] = 0
    inverted_data1 = np.fft.ifft(fft_values)
    fft_values[cutoff_index] = saved_high_freqs
    inverted_data2 = np.fft.ifft(fft_values)
    plt.figure()
    plt.plot(data[:t], label='original')
    plt.plot(inverted_data1[:t], label='inverted_data1')
    plt.plot(inverted_data2[:t], label='inverted_data2')
    plt.legend()
    plt.show()

def plot_fft3(data, inverted_data, t,title):
    '''
    验证频域去噪效果
    '''
    plt.figure()
    plt.plot(data[:t], label='original')
    plt.plot(inverted_data[:t], label='inverted_data')
    plt.title('Frequency cut='+str(title))
    plt.legend()
    plt.show()
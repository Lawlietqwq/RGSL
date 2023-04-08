import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====热力图
from matplotlib.ticker import FormatStrFormatter

def heat_graph():
    data = np.load("../data/rpsdata/rps_data.npy")[:,:,3]  # 读取数据
    up_limit = pd.read_csv("../util/up_limit_dict.csv", index_col=0)
    df = pd.DataFrame(data, columns=up_limit.columns, index=up_limit.index)
    df = df.iloc[-31:]
    df = df.astype(int)
    df = pd.DataFrame(df.values.T,columns=df.index,index=df.columns)

    f, ax = plt.subplots(figsize=(12, 9))


    # 返回皮尔逊积矩相关系数
    sns.set(font_scale=1.25)
    hm = sns.heatmap(df,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt="d",
                     vmin=0,  # 刻度阈值
                     vmax=100,
                     linewidths=.5,
                     cmap="RdPu",  # 刻度颜色
                     annot_kws={"size": 10},
                     xticklabels=5,
                     yticklabels=True)  # seaborn.heatmap相关属性
    # 解决中文显示问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.ylabel(fontsize=15,)
    # plt.xlabel(fontsize=15)
    plt.title("RPS time series of stock sectors", fontsize=20)
    plt.show()

def bar_graph():
    data = pd.read_csv('../model/fct_ic.csv', index_col=0)

    legend_name = data.columns
    size = len(legend_name)

    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x_labels = data.index
    x = np.arange(0,12,4)

    # 有a/b/c三种类型的数据，n设置为3
    total_width = 3.5
    # 每种类型的柱状图宽度
    width = total_width / size
    plt.xticks(x, x_labels)

    # 重新设置x轴的坐标
    # x = x - (total_width - width) / 2
    # print(x)

    # 画柱状图
    for i in range(size):
        plt.bar(x+(i-size/2)*width, list(data[legend_name[i]]), width=width, label=legend_name[i])
    # 显示图例
    plt.legend()
    # 显示柱状图
    plt.show()

bar_graph()
import json
from collections import deque
# from os import pread
from tkinter.font import names
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tushare as ts
import os

TUSHARE_TOKEN = 'ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
scaler = StandardScaler()

json_file_path = '../data/rpsdata/'
json_file_name = json_file_path + 'label_reli.json'
json_file_nameo = json_file_path + 'label_relio.json'


def process_data():
    rps_day = ['1', '3', '5', '10', '15', '20']
    rps_dict = {}
    with open(json_file_nameo, 'r', encoding='utf-8') as f:
        loads = json.loads(f.read())
        names = loads['10']['name']
        # print(names)
        for n in rps_day:
            rps_list = []
            for name in names:
                rps_list.extend(loads[n][name])
            rps_dict['rps' + n] = rps_list
    with open(json_file_name, 'r', encoding='utf-8') as f:
        loads1 = json.loads(f.read())
        rps_list = []
        for name in names:
            rps_list.extend(loads1['1'][name])
        rps_dict['label'] = rps_list
    df = pd.DataFrame(data=rps_dict)
    #  数据处理
    # print(df)
    df.dropna(inplace=True)
    # df.sort_index(inplace=True)
    # df=df.drop([0])
    return df, names


def stock_price_lstm_Data_precesing(df, mem_his_days, pre_days):
    # 创建一个队列，先进先出
    deq = deque(maxlen=mem_his_days)
    sca_X = scaler.fit_transform(df.iloc[:, :-1])
    # print(sca_X)

    X = []
    for i in sca_X:
        deq.append(list(i))
        if len(deq) == mem_his_days:
            X.append(list(deq))
    # 记录最后几条数据
    X_lately = X[-pre_days:]
    X = X[:-pre_days]
    y = df['label'].values[mem_his_days - 1:-pre_days]

    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    # print(X)
    # print(y)
    return X, y, X_lately


sw_l2 = pro.index_classify(level='L2', src='SW2021')
sw = sw_l2.loc[:, ['industry_name', 'index_code']].set_index('industry_name')
# print(sw)

# 提取指数涨幅序列
stocks_price = pd.read_csv('./pricenew.csv')[0:170]
print(stocks_price)

df, names = process_data()
code_lists = []
for name in names:
    code_lists.append(sw.loc[name]['index_code'])
# code_lists
# print(code_lists)
# df_code_list=pd.DataFrame(data=code_lists)
# df_code_list.to_csv('test.csv')
for index, d in df.iterrows():
    if index % 170 == 0:
        csv_file = df.iloc[index:index + 170, :]
        file_url = '../data/rpsdata/sequences5/'
        if os.path.exists(file_url):
            pass
        else:
            os.makedirs(file_url)

        num = int(index / 170)
        csv_file['price_change'] = list(stocks_price[code_lists[num]][0:170])
        csv_file = csv_file[['price_change', 'rps1', 'rps3', 'rps5', 'rps10', 'rps15', 'rps20', 'label']]
        # print(csv_file)

        csv_file.to_csv(file_url + code_lists[num] + '.csv')
import json
from collections import deque
# from os import pread
from tkinter.font import names
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tushare as ts
import os
import time
from jqdatasdk import *

TUSHARE_TOKEN = 'ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
scaler = StandardScaler()

json_file_path='../data/nasdaq/'
# json_file_name=json_file_path+'label_reli.json'
# json_file_nameo=json_file_path+'label_relio.json'
json_file_nameo=json_file_path+'p_label_reli.json'
json_file_name=json_file_path+'p_label_reli.json'




def process_data():
    rps_day=['1','3','5','10','15','20']
    rps_dict={}
    with open(json_file_nameo,'r',encoding='utf-8') as f :
        loads = json.loads(f.read())
        names=loads['10']['name']
        date_list = loads['1']['time']
        # print(names)
        for n in rps_day:
            rps_list=[]
            for name in names:
                rps_list.extend(loads[n][name])
            rps_dict['rps'+n]=rps_list

    with open(json_file_name,'r',encoding='utf-8') as f :
        loads1 = json.loads(f.read())
        rps_list=[]
        for name in names:
            rps_list.extend(loads1['1'][name])
        rps_dict['label']=rps_list
    df= pd.DataFrame(data=rps_dict)
    #  数据处理
    # print(df)
    df.dropna(inplace=True)
    # df.sort_index(inplace=True)
    # df=df.drop([0])
    return df,names,date_list

df, names, date_list = process_data()
code_lists = names
# 1742, 144
num = len(code_lists)
seq = len(date_list)
print(seq)
rps1 = df['rps1'].to_numpy().reshape(num, seq).T
rps3 = df['rps3'].to_numpy().reshape(num, seq).T
rps5 = df['rps5'].to_numpy().reshape(num, seq).T
rps10 = df['rps10'].to_numpy().reshape(num, seq).T
rps15 = df['rps15'].to_numpy().reshape(num, seq).T
rps20 = df['rps20'].to_numpy().reshape(num, seq).T
pct_change = pd.read_csv('../data/nasdaq/pct_change.csv', index_col=0).iloc[20:].to_numpy()
close_price = pd.read_csv('../data/nasdaq/close_price.csv', index_col=0).iloc[20:].to_numpy()
open_price = pd.read_csv('../data/nasdaq/open_price.csv', index_col=0).iloc[20:].to_numpy()
turnover_fct = pd.read_csv('../data/nasdaq/turnover.csv', index_col=0).iloc[20:].to_numpy()
low_price = pd.read_csv('../data/nasdaq/low_price.csv', index_col=0).iloc[20:].to_numpy()
volume = pd.read_csv('../data/nasdaq/volume.csv', index_col=0).iloc[20:].to_numpy()
high_price = pd.read_csv('../data/nasdaq/high_price.csv', index_col=0).iloc[20:].to_numpy()
up_limit_fct = pd.read_csv('../data/nasdaq/up_limit_dict.csv', index_col=0).to_numpy()
arr = np.zeros([seq, num, 9])
arr[...,0] = rps1
arr[...,1] = rps3
arr[...,2] = rps5
arr[...,3] = rps10
arr[...,4] = rps15
arr[...,5] = rps20
arr[...,6] = pct_change
arr[...,7] = up_limit_fct
arr[...,8] = close_price
# arr[...,9] = turnover_fct
# # # # all fct
# arr[...,10] = volume
# arr[...,11] = low_price
# arr[...,12] = high_price
# arr[...,13] = open_price
# arr[...,0] = pct_change
# arr[...,1] = close_price
# arr[...,2] = up_limit_fct
# # all fct
# arr[...,3] = turnover_fct
# arr[...,4] = low_price
# arr[...,5] = high_price
# arr[...,6] = open_price
# arr[...,7] = volume
# arr[...,8] = rps1
# arr = arr/100
print(arr)
np.save('../data/nasdaq/rps_data.npy', arr)
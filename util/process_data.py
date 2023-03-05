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

TUSHARE_TOKEN = 'ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
scaler = StandardScaler()


json_file_path='../data/rpsdata/'
json_file_name=json_file_path+'label_reli.json'
json_file_nameo=json_file_path+'label_relio.json'


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
def stock_price_lstm_Data_precesing(df,mem_his_days,pre_days):
    #创建一个队列，先进先出
    deq=deque(maxlen=mem_his_days)
    sca_X=scaler.fit_transform(df.iloc[:,:-1])
    # print(sca_X)

    X=[]
    for i in sca_X:
        deq.append(list(i))
        if len(deq)==mem_his_days:
            X.append(list(deq))
    # 记录最后几条数据
    X_lately = X[-pre_days:]
    X = X[:-pre_days]
    y=df['label'].values[mem_his_days-1:-pre_days]

    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    # print(X)
    # print(y)
    return X,y,X_lately
sw_l2 = pro.index_classify(level='L2', src='SW2021')
sw=sw_l2.loc[:,['industry_name','index_code']].set_index('industry_name')
# print(sw)

#提取指数涨幅序列
# stocks_price=pd.read_csv('pricenew.csv')[0:170]
# print(stocks_price)

#
# data = pd.read_csv('up_limit_dict.csv')
# data = data.fillna(0)

df, names, date_list = process_data()
code_lists=[]
# fct = pd.DataFrame(index=date_list)
for name in names:
    code_lists.append(sw.loc[name]['index_code'])

# 板块内涨停
def up_limit(code_lists, date_list):
    fct = pd.DataFrame(index=date_list)
    up_limit_dict = []
    thre = 0
    for industry_code in code_lists:
        df_index = pro.index_member(index_code=industry_code, is_new="Y")
        stock_list = df_index.con_code.unique()
        up_limit_list = pd.Series(0, index=range(len(date_list)))
        for code in stock_list:
            thre = thre + 1
            tmp = pro.stk_limit(ts_code=code, start_date='20211002', end_date=date_list[0], fields='ts_code,trade_date,pre_close,up_limit')
            if len(tmp)==0:
                continue
            pre = tmp.shift(1).pre_close
            tmp['pre_close'] = pre
            tmp = tmp[tmp.trade_date >= date_list[-1]]
            # up_limit = tmp['up_limit']
            # tmp['pre_close'] = pro.daily(ts_code=code, start_date=date_list[-1], end_date=date_list[0]).pre_close
            up_limit_list += (tmp.pre_close >= tmp.up_limit) + 0
            if thre == 380:
                time.sleep(60)
                thre = 0
        # np.save('../data/rpsdata/up_limit/{}'.format(industry_code), up_limit_list)
        # up_limit_dict.extend(up_limit_list)
        up_limit_list.index = date_list
        fct[industry_code] = up_limit_list
    # rps_dict['up_limit'] = up_limit_dict
    print(up_limit_dict)
    fct = fct.fillna(0)
    fct.to_csv('up_limit_dict.csv')

# 换手率
def turnover(code_lists ,date_list):
    turnover_fct = pd.DataFrame(index=date_list)
    for industry_code in code_lists:
        df_index = pro.index_member(index_code=industry_code, is_new="Y")
        stock_list = df_index.con_code.unique()
        industry_df = pd.DataFrame()
        for code in stock_list:
            single_df = pro.daily_basic(ts_code=code, start_date=date_list[-1],
                                 end_date=date_list[0], fields='ts_code,trade_date,turnover_rate,circ_mv')

            industry_df = industry_df.append(single_df)
        industry_df = list(industry_df.groupby('trade_date'))
        industry_turn_over = []
        for idx in range(len(industry_df)):
            tmp = industry_df[idx][1]
            avg_mkt = tmp['circ_mv']/tmp['circ_mv'].sum()
            tmp['avg_mkt'] = avg_mkt
            tmp['turn_over'] = avg_mkt * tmp['turnover_rate']
            turn_over = tmp['turn_over'].fillna(0).sum()
            industry_turn_over.append(turn_over)
        # industry_turn_over = list(reversed(industry_turn_over))
        turnover_fct[industry_code] = industry_turn_over
        # turnover_fct.insert(column=industry_code, value=industry_turn_over)
    turnover_fct.to_csv('turnover.csv')

# pe and pb
def pe(code_lists ,date_list):
    pe_fct = pd.DataFrame(index=date_list)
    pb_fct = pd.DataFrame(index=date_list)
    idx = 0
    for industry_code in code_lists:
        df1 = pro.sw_daily(ts_code=industry_code, start_date=date_list[-1], end_date=date_list[0])
        pe_fct[industry_code] = df1['pe']
        pb_fct[industry_code] = df1['pb']
        if idx==100:
            time.sleep(60)
            idx=0

    pe_fct.to_csv('pe.csv')
    pb_fct.to_csv('pb.csv')

# 行业指数涨幅
def industry_index(code_lists, date_list):
    df_index = pd.DataFrame(index=date_list)
    for industry_code in code_lists:
        tmp = pro.sw_daily(ts_code=industry_code, start_date=date_list[-1], end_date=date_list[0])
        df_index[industry_code] = list(tmp['pct_change'])
    df_index.to_csv('pct_change.csv')
    print("industry_index完成")

# #配对相关性
# def pd_sim(code_lists ,date_list):
#     fct = pd.DataFrame(index=date_list)
#     for industry_code in code_lists:
#         df_index = pro.index_member(index_code=industry_code, is_new="Y")
#         stock_list = df_index.con_code.unique()
#         industry_df = pd.DataFrame()
#         nums = len(stock_list)
#         for code in stock_list:
#
#     pe_fct.to_csv('turnover.csv')
print('______________')
# industry_index(code_lists, date_list)
pe(code_lists, date_list)
# up_limit(code_lists,date_list)
# turnover(code_lists,date_list)
# np.save('up_limit_dict',fct)
#
# # df1 = pro.sw_daily(ts_code=code_lists, start_date="20211102", end_date="20220812")[
# #       0:134]
#
#
# num = len(code_lists)
# rps1 = df['rps1'].to_numpy().reshape(170, num)
# rps3 = df['rps3'].to_numpy().reshape(170, num)
# rps5 = df['rps5'].to_numpy().reshape(170, num)
# rps10 = df['rps10'].to_numpy().reshape(170, num)
# rps15 = df['rps15'].to_numpy().reshape(170, num)
# rps20 = df['rps20'].to_numpy().reshape(170, num)
# arr = np.zeros([170, 124, 7])
# arr[...,0] = rps1
# arr[...,1] = rps3
# arr[...,2] = rps5
# arr[...,3] = rps10
# arr[...,4] = rps15
# arr[...,5] = rps20
# arr = arr/100
# arr[...,6] = up_limit_dict.to_numpy().reshape(170, num)
# print(arr)
# np.save('rps_data', arr)
# for index,d in df.iterrows():
#     if index%170==0:
#         csv_file=df.iloc[index:index+170,:]
#         file_url='../data/rpsdata/sequences7/'
#         if os.path.exists(file_url):
#             pass
#         else:
#             os.makedirs(file_url)
#
#         num=int(index/170)
#         csv_file['price_change']=list(stocks_price[code_lists[num]][0:170])
#         # csv_file=csv_file[['price_change','rps1','rps3','rps5','rps10','rps15','rps20','label']]
#         csv_file=csv_file[code_lists]
#         # print(csv_file)
#
#         csv_file.to_csv(file_url+code_lists[num]+'.csv')
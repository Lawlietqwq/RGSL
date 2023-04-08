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
auth('18974988801', 'Bigdata12345678')

TUSHARE_TOKEN = 'ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
scaler = StandardScaler()


json_file_path='../data/rpsdata/'
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
# sw_l2 = pro.index_classify(level='L2', src='SW2021')
# sw=sw_l2.loc[:,['industry_name','index_code']].set_index('industry_name')
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
# for name in names:
    # code_lists.append(sw.loc[name]['index_code'])
code_lists = names
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
            if thre == 300:
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

def up_limit2(code_lists, date_list):
    fct = pd.DataFrame(index=date_list)
    up_limit_dict = []
    for industry_code in code_lists:
        df_index = get_industry_stocks(industry_code, date='2021-12-10')
        up_limit_list = pd.Series(0, index=date_list)
        for code in df_index:
            industry_df = get_price(code, start_date='2005-01-04', end_date='2021-12-10', frequency='daily',
                                    fields=['close', 'high_limit', 'paused'])
            industry_df = industry_df[industry_df.index >= date_list[0]]
            industry_df.index = date_list
            # up_limit = tmp['up_limit']
            # tmp['pre_close'] = pro.daily(ts_code=code, start_date=date_list[-1], end_date=date_list[0]).pre_close
            up_limit_list += ((industry_df.close >= industry_df.high_limit) | (industry_df.paused)) + 0
        # np.save('../data/rpsdata/up_limit/{}'.format(industry_code), up_limit_list)
        # up_limit_dict.extend(up_limit_list)
        fct[industry_code] = up_limit_list
    # rps_dict['up_limit'] = up_limit_dict
    print(up_limit_dict)
    fct = fct.fillna(0)
    fct.to_csv('up_limit_dict.csv')

# 换手率
## tushare
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


def turn_over(code_lists ,date_list):
    turnover_fct = pd.DataFrame(index=date_list)
    for industry_code in code_lists:
        df_index = get_industry_stocks(industry_code, date=date_list[-1])
        industry_df = pd.DataFrame()
        for code in df_index:
            q = query(
                valuation.turnover_ratio,
                valuation.circulating_market_cap
            ).filter(
                valuation.code == code
            )

            single_df = get_fundamentals_continuously(q, end_date=date_list[-1], count=len(date_list))
            # single_df = single_df[single_df.day >= date_list[0]]
            industry_df = industry_df.append(single_df)

        industry_df = list(industry_df.groupby('day'))
        industry_turn_over = []
        for idx in range(len(industry_df)):
            tmp = industry_df[idx][1]
            avg_mkt = tmp['circulating_market_cap']/tmp['circulating_market_cap'].sum()
            tmp['avg_mkt'] = avg_mkt
            tmp['turn_over'] = avg_mkt * tmp['turnover_ratio']
            turn_over = tmp['turn_over'].fillna(0).sum()
            industry_turn_over.append(turn_over)
        # industry_turn_over = list(reversed(industry_turn_over))
        turnover_fct[industry_code] = industry_turn_over
        # turnover_fct.insert(column=industry_code, value=industry_turn_over)
    turnover_fct.to_csv('turn_over.csv')

# pe and pb
def pe(code_lists ,date_list):
    pe_fct = pd.DataFrame(index=date_list)
    pb_fct = pd.DataFrame(index=date_list)
    idx = 0
    date = date_list[-1].replace('-','')
    for industry_code in code_lists:

        df1 = pro.sw_daily(ts_code=industry_code+'.SI', end_date=date)
        pe_fct[industry_code] = list(df1['pe'])
        pb_fct[industry_code] = list(df1['pb'])
        if idx==100:
            time.sleep(60)
            idx=0

    pe_fct.to_csv('pe.csv')
    pb_fct.to_csv('pb.csv')

# 行业指数涨幅
def industry_index(code_lists, date_list):
    df_index = pd.DataFrame(index=date_list)
    for industry_code in code_lists:
        df1 = finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code == industry_code,
                                                                      finance.SW1_DAILY_PRICE.date >= date_list[0],
                                                                      finance.SW1_DAILY_PRICE.date <= '2020-08-19'))
        df2 = finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code == industry_code,
                                                                      finance.SW1_DAILY_PRICE.date >= '2020-08-20',
                                                                      finance.SW1_DAILY_PRICE.date <= date_list[-1]))
        tmp = pd.concat([df1, df2])
        # tmp = pro.sw_daily(ts_code=industry_code, start_date=date_list[-1], end_date=date_list[0])
        df_index[industry_code] = list(tmp['change_pct'])
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

#价格
def industry_price(code_lists ,date_list):
    df_index = pd.DataFrame(index=date_list)
    df_vol = pd.DataFrame(index=date_list)
    df_open = pd.DataFrame(index=date_list)
    df_high = pd.DataFrame(index=date_list)
    df_low = pd.DataFrame(index=date_list)
    for industry_code in code_lists:
        df1 = finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code == industry_code,
                                                                      finance.SW1_DAILY_PRICE.date >= date_list[0],
                                                                      finance.SW1_DAILY_PRICE.date <= '2020-08-19'))
        df2 = finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code == industry_code,
                                                                      finance.SW1_DAILY_PRICE.date >= '2020-08-20',
                                                                      finance.SW1_DAILY_PRICE.date <= date_list[-1]))
        tmp = pd.concat([df1, df2])
        # tmp = pro.sw_daily(ts_code=industry_code, start_date=date_list[-1], end_date=date_list[0])
        # df_index[industry_code] = list(tmp['close'])
        df_index[industry_code] = list(tmp['close'])
        df_vol[industry_code] = list(tmp['volume'])
        df_open[industry_code] = list(tmp['open'])
        df_high[industry_code] = list(tmp['high'])
        df_low[industry_code] = list(tmp['low'])
    df_index.to_csv('close_price.csv')
    df_vol.to_csv('volume.csv')
    df_open.to_csv('open_price.csv')
    df_high.to_csv('high_price.csv')
    df_low.to_csv('low_price.csv')
    print("close_price完成")

print('______________')
industry_price(code_lists,date_list)
# industry_index(code_lists, date_list)
# pe(code_lists, date_list)
# up_limit2(code_lists,date_list)
# turn_over(code_lists,date_list)
# np.save('up_limit_dict',fct)
#
# # df1 = pro.sw_daily(ts_code=code_lists, start_date="20211102", end_date="20220812")[
# #       0:134]
#
#

# 4097
num = len(code_lists)
seq = len(date_list)
print(seq)
rps1 = df['rps1'].to_numpy().reshape(num, seq).T
rps3 = df['rps3'].to_numpy().reshape(num, seq).T
rps5 = df['rps5'].to_numpy().reshape(num, seq).T
rps10 = df['rps10'].to_numpy().reshape(num, seq).T
rps15 = df['rps15'].to_numpy().reshape(num, seq).T
rps20 = df['rps20'].to_numpy().reshape(num, seq).T
pct_change = pd.read_csv('../util/pct_change.csv', index_col=0).to_numpy()
close_price = pd.read_csv('../util/close_price.csv', index_col=0).to_numpy()
open_price = pd.read_csv('../util/open_price.csv', index_col=0).to_numpy()
turnover_fct = pd.read_csv('../util/turn_over.csv', index_col=0).to_numpy()
low_price = pd.read_csv('../util/low_price.csv', index_col=0).to_numpy()
volume = pd.read_csv('../util/volume.csv', index_col=0).to_numpy()
high_price = pd.read_csv('../util/high_price.csv', index_col=0).to_numpy()
up_limit_fct = pd.read_csv('../util/up_limit_dict.csv', index_col=0).to_numpy()
arr = np.zeros([seq, num, 8])
arr[...,0] = rps1
arr[...,1] = rps3
arr[...,2] = rps5
arr[...,3] = rps10
arr[...,4] = rps15
arr[...,5] = rps20
arr[...,6] = pct_change
arr[...,7] = close_price
arr[...,8] = up_limit_fct
# all fct
arr[...,9] = open_price
arr[...,10] = low_price
arr[...,11] = high_price
arr[...,12] = turnover_fct
arr[...,13] = volume
# arr = arr/100
print(arr)
np.save('../data/rpsdata/rps_data.npy', arr)


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
from util.section_industry import SectorPreprocessor
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
import csv
import akshare as ak
# auth('18974988801', 'Bigdata12345678')


TUSHARE_TOKEN = 'ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
scaler = StandardScaler()

# industry_path = '../data/nasdaq/industry_relation_Nasdaq.json'
code_list_path = '../data/nasdaq/code_list.json'
stock_path = '../data/nasdaq/stock_list.json'
code_list = []
stock_relation = {}
with open(code_list_path, 'r', encoding='utf-8') as f:
    code_list = json.loads(f.read())
    code_list = list(code_list)
    print(code_list)

with open(stock_path, 'r', encoding='utf-8') as f:
    stock_relation = json.loads(f.read())

date_list = pro.us_tradecal(start_date='20160301', end_date='20230301')
date_list = list(date_list[date_list['is_open'] == 1]['cal_date'])
date_list = list(reversed(date_list))
date_list.remove('20220620')
print('{}天的股票'.format(len(date_list)))

mapping = {}
with open('../data/nasdaq/dongcai_list.json', 'r', encoding='utf-8') as f:
    mapping = json.loads(f.read())

# def getsectors(path='../data/sectors/'):
#     stock_relation = {}
#     start_date = int(date_list[0])
#     end_date = int(date_list[-1])
#     stock_num = 0
#     all_stock_list = pd.read_csv('../data/rpsdata/all_nasdaq_stock.csv', index_col=0)
#     all_stock_list = all_stock_list[(all_stock_list['list_date'] < start_date)]
#     all_stock_list = all_stock_list[
#         (np.isnan(all_stock_list['delist_date'])) | (all_stock_list['delist_date'] > end_date)]
#     all_stock_list = list(all_stock_list['ts_code'])
#     for i in range(147):
#         p = path + 'sector{}.csv'.format(i+1)
#         df = pd.read_csv(p)
#         stock_list = list(df['Symbol'])
#         for j in range(len(stock_list)-1,-1,-1):
#             stock = stock_list[j]
#             if stock not in all_stock_list:
#                 stock_list.remove(stock)
#         if len(stock_list) == 0:
#             print(i+1001)
#         else:
#             stock_relation[str(i+1001)] = stock_list
#         stock_num += len(stock_list)
#     print("总共{}个股票".format(stock_num))
#     with open('../data/nasdaq/stock_list.json', 'w+') as f:
#         json.dump(stock_relation, f)
#     with open('../data/nasdaq/code_list.json', 'w+') as f:
#         json.dump(list(stock_relation.keys()), f)

#筛选需要的在售股票
def getsectors(path='../data/sectors/'):
    stock_relation = {}
    start_date = int(date_list[0])
    end_date = int(date_list[-1])
    stock_num = 0
    all_stock_list = pd.read_csv('../data/rpsdata/all_nasdaq_stock.csv', index_col=0)
    mapping_keys = mapping.keys()
    all_stock_list = all_stock_list[(all_stock_list['list_date'] < start_date)]
    all_stock_list = all_stock_list[
        (np.isnan(all_stock_list['delist_date'])) | (all_stock_list['delist_date'] > end_date)]
    all_stock_list = list(all_stock_list['ts_code'])
    for i in range(147):
        p = path + 'sector{}.csv'.format(i+1)
        df = pd.read_csv(p)
        stock_list = list(df['Symbol'])
        for j in range(len(stock_list)-1,-1,-1):
            stock = stock_list[j]
            if stock not in all_stock_list or stock not in mapping_keys:
                stock_list.remove(stock)
        if len(stock_list) == 0:
            print(i+1001)
        else:
            stock_relation[str(i+1001)] = stock_list
        stock_num += len(stock_list)
    print("总共{}个股票".format(stock_num))
    with open('../data/nasdaq/stock_list.json', 'w+') as f:
        json.dump(stock_relation, f)
    with open('../data/nasdaq/code_list.json', 'w+') as f:
        json.dump(list(stock_relation.keys()), f)

# getsectors()

# 所有纳斯达克股票
def all_stock_to_csv():
    all_stock_list = pd.DataFrame()
    for i in range(10):
        df = pro.us_basic(offset=i*6000)
        if len(df) != 0:
            all_stock_list = pd.concat([all_stock_list, df])
        else: break
    all_stock_list.index = [j for j in range(len(all_stock_list))]
    all_stock_list.to_csv('../data/rpsdata/all_nasdaq_stock.csv')


# all_stock_to_csv()

def drop_delist(stock_relation, code_list, date_list):
    start_date = int(date_list[0])
    end_date = int(date_list[-1])
    stock_num = 0
    all_stock_list = pd.read_csv('../data/rpsdata/all_nasdaq_stock.csv', index_col=0)
    all_stock_list = all_stock_list[(all_stock_list['list_date']<start_date)]
    all_stock_list = all_stock_list[(np.isnan(all_stock_list['delist_date'])) | (all_stock_list['delist_date']>end_date)]
    all_stock_list = list(all_stock_list['ts_code'])
    for code in code_list:
        stock_list = stock_relation.get(code)
        for i in range(len(stock_list)-1,-1,-1):
            stock = stock_list[i]
            if stock not in all_stock_list:
                stock_list.remove(stock)
        if len(stock_list) == 0:
            print(code)
        else:
            stock_relation[code] = stock_list
        stock_num += len(stock_list)
    print("总共{}个股票".format(stock_num))
    with open('../data/rpsdata/new_nasdaq_stock.json', 'w+') as f:
        json.dump(stock_relation, f)

def cal_sector():
    start_date = date_list[0]
    end_date = date_list[-1]
    new_date_list = date_list
    turnover_fct = pd.DataFrame(index=new_date_list)
    df_close = pd.DataFrame(index=new_date_list)
    df_open = pd.DataFrame(index=new_date_list)
    df_high = pd.DataFrame(index=new_date_list)
    df_low = pd.DataFrame(index=new_date_list)
    df_vol = pd.DataFrame(index=new_date_list)
    for industry_code in code_list:
        industry_df = pd.DataFrame()
        stock_list = stock_relation.get(industry_code)
        for stock_code in stock_list:
            code = mapping.get(stock_code)
            single_df = ak.stock_us_hist(symbol=code, period='daily', start_date=start_date,
                                                          end_date=end_date, adjust="")
            single_df['日期'] = single_df['日期'].replace('-', '')
            single_df['mkt'] = single_df['成交量']/single_df['换手率'] * single_df['收盘']
            single_df['mkt'] = single_df['mkt'].replace([np.inf, -np.inf], np.nan)
            single_df['mkt'] = single_df['mkt'].fillna(0)
            industry_df = pd.concat([industry_df, single_df])
        industry_df = list(industry_df.groupby('日期'))
        industry_close = []
        industry_open = []
        industry_high = []
        industry_low = []
        industry_turn_over = []
        industry_vol = []
        for idx in range(len(industry_df)):
            tmp = industry_df[idx][1]
            tmp_sum = tmp['mkt'].sum()
            avg_mkt = tmp['mkt']/tmp_sum
            tmp['avg_mkt'] = avg_mkt
            tmp['换手率'] = avg_mkt * tmp['换手率']
            tmp['收盘'] = avg_mkt * tmp['收盘']
            tmp['开盘'] = avg_mkt * tmp['开盘']
            tmp['最高'] = avg_mkt * tmp['最高']
            tmp['最低'] = avg_mkt * tmp['最低']
            turn_over = tmp['换手率'].fillna(0).sum()
            close = tmp['收盘'].fillna(0).sum()
            open = tmp['开盘'].fillna(0).sum()
            high = tmp['最高'].fillna(0).sum()
            low = tmp['最低'].fillna(0).sum()
            vol = tmp['成交量'].fillna(0).sum()
            industry_turn_over.append(turn_over)
            industry_close.append(close)
            industry_open.append(open)
            industry_high.append(high)
            industry_low.append(low)
            industry_vol.append(vol)

        print(industry_code)
        if len(industry_turn_over)!=1763:
            print('{}缺少数据'.format(industry_code))
            continue
        turnover_fct[industry_code] = industry_turn_over
        df_close[industry_code] = industry_close
        df_open[industry_code] = industry_open
        df_high[industry_code] = industry_high
        df_low[industry_code] = industry_low
        df_vol[industry_code] = industry_vol

    for row in range(1,turnover_fct.shape[0]):
        for col in range(turnover_fct.shape[1]):
            tmp = df_close.iloc[row-1,col]
            if df_close.iloc[row,col] == 0:
                df_close.iloc[row,col] = tmp
            if df_open.iloc[row,col] == 0:
                df_open.iloc[row,col] = tmp
            if df_high.iloc[row,col] == 0:
                df_high.iloc[row,col] = tmp
            if df_low.iloc[row,col] == 0:
                df_low.iloc[row,col] = tmp

    df_pct = (df_close - df_close.shift(1)) / df_close.shift(1)
    turnover_fct.iloc[1:].to_csv('../data/nasdaq/turnover.csv')
    df_pct.iloc[1:].to_csv('../data/nasdaq/pct_change.csv')
    df_close.iloc[1:].to_csv('../data/nasdaq/close_price.csv')
    df_open.iloc[1:].to_csv('../data/nasdaq/open_price.csv')
    df_high.iloc[1:].to_csv('../data/nasdaq/high_price.csv')
    df_low.iloc[1:].to_csv('../data/nasdaq/low_price.csv')
    df_vol.iloc[1:].to_csv('../data/nasdaq/volume.csv')

# cal_sector()

def up_limit():
    start_date = date_list[21]
    end_date = date_list[-1]
    new_date_list = date_list[21:]
    fct = pd.DataFrame(index=new_date_list)
    up_limit_dict = []
    for industry_code in code_list:
        up_limit_list = pd.Series(0, index=new_date_list)
        stock_list = stock_relation.get(industry_code)
        for stock_code in stock_list:
            code = mapping.get(stock_code)
            industry_df = ak.stock_us_hist(symbol=code, period='daily', start_date=start_date,
                                                          end_date=end_date, adjust="")
            industry_df['日期'] = industry_df['日期'].replace(to_replace={'-':''},regex=True)
            # tmp = industry_df['日期']
            # for i in range(len(new_date_list)):
            #     if tmp.iloc[i].replace('-','')!=new_date_list[i]:
            #         print(tmp.iloc[i])
            #         print(new_date_list[i])
            #         print(i)
            industry_df =  industry_df.set_index('日期')
            print(stock_code)
            up_limit_list = up_limit_list.add((industry_df['涨跌幅'] >= 0.1) + 0, fill_value = 0)
        fct[industry_code] = up_limit_list

    print(up_limit_dict)
    fct = fct.fillna(0)
    fct.to_csv('../data/nasdaq/up_limit_dict.csv')

# up_limit()


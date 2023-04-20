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

date_list = pro.us_tradecal(start_date='20100301', end_date='20230301')
date_list = list(date_list[date_list['is_open'] == 1]['cal_date'])
date_list = list(reversed(date_list))
print('{}天的股票'.format(len(date_list)))

def getsectors(path='../data/sectors/'):
    stock_relation = {}
    start_date = int(date_list[0])
    end_date = int(date_list[-1])
    stock_num = 0
    all_stock_list = pd.read_csv('../data/rpsdata/all_nasdaq_stock.csv', index_col=0)
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
            if stock not in all_stock_list:
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

#筛选需要的在售股票
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

# all_stock_to_csv()
# drop_delist(stock_relation, code_list, date_list)

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
    turnover_fct = pd.DataFrame(index=date_list)
    df_close = pd.DataFrame(index=date_list)
    df_open = pd.DataFrame(index=date_list)
    df_high = pd.DataFrame(index=date_list)
    df_low = pd.DataFrame(index=date_list)
    df_vol = pd.DataFrame(index=date_list)
    for industry_code in code_list:
        industry_df = pd.DataFrame()
        stock_list = stock_relation.get(industry_code)
        for stock_code in stock_list:
            single_df = pro.us_daily(ts_code=stock_code,start_date=start_date, end_date=end_date)
            time.sleep(30)
            industry_df = pd.concat([industry_df, single_df])
        industry_df = list(industry_df.groupby('trade_date'))
        industry_close = []
        industry_open = []
        industry_high = []
        industry_low = []
        industry_turn_over = []
        industry_vol = []
        for idx in range(len(industry_df)):
            tmp = industry_df[idx][1]
            avg_mkt = tmp['total_mv']/tmp['total_mv'].sum()
            tmp['avg_mkt'] = avg_mkt
            tmp['turn_over'] = avg_mkt * tmp['turnover_ratio']
            tmp['close'] = avg_mkt * tmp['close']
            tmp['open'] = avg_mkt * tmp['open']
            tmp['high'] = avg_mkt * tmp['high']
            tmp['low'] = avg_mkt * tmp['low']
            tmp['vol'] = avg_mkt * tmp['vol']
            turn_over = tmp['turn_over'].fillna(0).sum()
            close = tmp['close'].fillna(0).sum()
            open = tmp['open'].fillna(0).sum()
            high = tmp['high'].fillna(0).sum()
            low = tmp['low'].fillna(0).sum()
            vol = tmp['vol'].fillna(0).sum()
            industry_turn_over.append(turn_over)
            industry_close.append(close)
            industry_open.append(open)
            industry_high.append(high)
            industry_low.append(low)
            industry_vol.append(vol)

        turnover_fct[industry_code] = industry_turn_over
        df_close[industry_code] = industry_close
        df_open[industry_code] = industry_open
        df_high[industry_code] = industry_high
        df_low[industry_code] = industry_low
        df_vol[industry_code] = industry_vol

    df_pct = (df_close - df_close.shift(1)) / df_close.shift(1)
    turnover_fct.iloc[1:].to_csv('../data/nasdaq/turnover.csv')
    df_pct.iloc[1:].to_csv('../data/nasdaq/pct_change.csv')
    df_close.iloc[1:].to_csv('../data/nasdaq/close_price.csv')
    df_open.iloc[1:].to_csv('../data/nasdaq/open_price.csv')
    df_high.iloc[1:].to_csv('../data/nasdaq/high_price.csv')
    df_low.iloc[1:].to_csv('../data/nasdaq/low_price.csv')
    df_vol.iloc[1:].to_csv('../data/nasdaq/volume.csv')

cal_sector()
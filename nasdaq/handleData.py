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
# code_list_path = '../data/nasdaq/code_list.csv'
# stock_path = '../data/nasdaq/stock_list.json'
# code_list = []
# stock_relation = {}
# with open(code_list_path, 'r', encoding='utf-8') as f:
#     code_list = json.loads(f.read())
#     code_list = list(code_list)
#     print(code_list)
#
# with open(stock_path, 'r', encoding='utf-8') as f:
#     stock_relation = json.loads(f.read())
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
getsectors()

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







def getdatas(date_length):
    length=len(code_lists)
    datas={}
    times=[]
    count=1
    # tscode=list(df['index_code'])
    tscode=code_lists
    # for i,tscode in enumerate(df['index_code']):
    while tscode:
        print("代码{}，这是第{}个，还剩{}个".format(tscode[0],count,length-count))
        try:
            #我这里的周期是一年，one_year_time 是选中时间一年前的日期
            # df1 = pro.sw_daily(ts_code=tscode[0], start_date="20211102", end_date="20220812", fields='ts_code,trade_date,close')[0:134]
            # df1 = pro.sw_daily(ts_code=tscode[0], start_date="20211102", end_date="20230228", fields='ts_code,trade_date,close')
            df1=finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code==tscode[0],
                                                                        finance.SW1_DAILY_PRICE.date >= '2005-01-04',
                                                                        finance.SW1_DAILY_PRICE.date <= '2021-12-10'))
            print("len:",len(df1))
            if len(df1)<date_length:
                print('{}指数不存在或者数据量太小'.format(tscode[0]))
                tscode.pop(0)
                continue
            datas[tscode[0]]=df1['close']
            #如果提交成功，则将第一个代码剔除，如果失败进入异常，之后循环将会重新提交
            tscode.pop(0)
            count=count+1

            #获取时间序列作为x轴
            if len(times)==0:
                times=df1['date']
            # time.sleep(1)
        except Exception as e:
            print('出现提交错误，之后将会重新提交')
            # tscode.append(tscode[0])
            # count=count-1
    all_stocks=pd.DataFrame(data=datas)
    all_stocks.index=times
    return all_stocks


#计算RPS
def getRPS(end_time,all_stocks):
    arr_N = [20,15,10,5,3,1]  # 计算周期
    # print(end_time)
    all_stocks=all_stocks[all_stocks.index<=end_time]
    # print(all_stocks)

    dict_rps = {}
    for n in arr_N:
        # rank
        # print(len(all_stocks))
        # s_rank = (all_stocks.iloc[0]/all_stocks.iloc[n]).sort_values(ascending=False).rank(method='dense')
        # s_rank = (all_stocks.iloc[0]/all_stocks.iloc[n]).rank(method='dense')
        s_rank = (all_stocks.iloc[-1]/all_stocks.iloc[-1-n]).rank(method='dense')
        # 归一
        s_rps = (100*(s_rank - s_rank.min()))/(s_rank.max()-s_rank.min())
        # 写入字典
        # s_rps=s_rps.sort_index()
        dict_rps[n] = s_rps.to_dict()
    # print(dict_rps)
    # rps = dict_rps[10]
    # rps[rps > 99]
    return dict_rps

# 将数据转化为热力图数据
def get_reli_datatype():
    industrys_list=[]
    times_list=[]
    dicts1={}
    dicts3={}
    dicts5={}
    dicts10={}
    dicts15={}
    dicts20={}
    dicts={}
    with open('../data/rpsdata/p_label.json','r',encoding='utf-8') as f :
            loads = json.loads(f.read())
            times_list=loads['date']
            industrys_list = loads['code']
            # for ind in industrys_list:
            #     name=df.at[ind,'industry_name']
            #     names_list.append(name)
            # for name in names_list:
            for ind in industrys_list:
                    # name=df.at[ind,'industry_name']
                    price_list1=[]
                    price_list3=[]
                    price_list5=[]
                    price_list10=[]
                    price_list15=[]
                    price_list20=[]

                    for mytime in times_list:
                            price_list1.append(math.floor(loads['data'][mytime]['1'][ind]))
                            price_list3.append(math.floor(loads['data'][mytime]['3'][ind]))
                            price_list5.append(math.floor(loads['data'][mytime]['5'][ind]))
                            price_list10.append(math.floor(loads['data'][mytime]['10'][ind]))
                            price_list15.append(math.floor(loads['data'][mytime]['15'][ind]))
                            price_list20.append(math.floor(loads['data'][mytime]['20'][ind]))
                            # print(loads[mytime]['50'][ind])
                    # print(price_list10)
                    times_list=list(times_list)
                    dicts1['time']=times_list
                    dicts3['time']=times_list
                    dicts5['time']=times_list
                    dicts10['time']=times_list
                    dicts15['time']=times_list
                    dicts20['time']=times_list
                    dicts1['name']=industrys_list
                    dicts3['name']=industrys_list
                    dicts5['name']=industrys_list
                    dicts10['name']=industrys_list
                    dicts15['name']=industrys_list
                    dicts20['name']=industrys_list
                    dicts1[ind]=price_list1
                    dicts3[ind]=price_list3
                    dicts5[ind]=price_list5
                    dicts10[ind]=price_list10
                    dicts15[ind]=price_list15
                    dicts20[ind]=price_list20
            dicts[1]=dicts1
            dicts[3]=dicts3
            dicts[5]=dicts5
            dicts[10]=dicts10
            dicts[15]=dicts15
            dicts[20]=dicts20

    with open('../data/rpsdata/p_label_reli.json','w+') as f :
        json.dump(dicts,f)

# 获取json格式的文件
def get_json_lists():
    # 获取交易日列表，获取需要更新的日期列表
    # 因为我用的申万数据是新的，数据最新的时间是20210101，而且为什么必须使用[50:]，为了使RPS50可以用上
    # day_list = pro.trade_cal(start_date='20050104', end_date='20211210')
    # day_list = list(day_list[day_list['is_open'] == 1]['cal_date'])
    day_list = []
    daylist = get_trade_days(start_date="2005-01-04",end_date="2021-12-10")
    for date in daylist:
        day_list.append(date.strftime('%Y-%m-%d'))
    get_day_list = day_list[20:]
    length=len(get_day_list)
    # print(get_day_list)
    n=len(get_day_list)
    #获取一段时间的所有的行业数据
    all_stocks=getdatas(length)
    all_stocks.index = day_list
    print(all_stocks)
    # #计算价格
    # stocks_price = all_stocks/all_stocks.shift(-1)-1
    # stocks_price = stocks_price.fillna(0)
    #
    # stocks_price.to_csv('price.csv')
    # print(stocks_price)
    #计算连续一段时间的所有行业RPS值
    dict_rpss={}
    for i in range(n):
        endday=get_day_list[i]
        print("时间:{}这是第{}个，还剩{}个".format(endday,i,length-i+1))
        dict_rps=getRPS(endday,all_stocks)
        dict_rpss[endday]=dict_rps
    # print(dict_rpss)
    dist = {}
    dist['data'] = dict_rpss
    dist['date'] = get_day_list
    dist['code'] = list(all_stocks.columns)
    with open('../data/rpsdata/p_label.json','w+') as f :
        json.dump(dist,f)
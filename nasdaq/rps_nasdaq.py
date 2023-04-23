
# 获取行业RPS数值
#先引入后面可能用到的library
# from pexpect import ExceptionPexpect
import tushare as ts
import pandas as pd
# import matplotlib.pyplot as plt
import time,datetime
import math
# from tools import get_date_bef,get_day_list
# %matplotlib inline
import json
import csv
import akshare as ak

TUSHARE_TOKEN = 'ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

code_list_path = '../data/nasdaq/code_list.json'
stock_path = '../data/nasdaq/stock_list.json'
close_path = '../data/nasdaq/close_price.csv'
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
date_list.remove('20220620')
date_list = list(reversed(date_list))[1:]
print('{}天的股票'.format(len(date_list)))

close_price = pd.read_csv(close_path, index_col=0)


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
    with open('../data/nasdaq/p_label.json','r',encoding='utf-8') as f :
            loads = json.loads(f.read())
            times_list=loads['date']
            times_list = list(map(str, times_list))
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

    with open('../data/nasdaq/p_label_reli.json','w+') as f :
        json.dump(dicts,f)

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
    with open('../data/nasdaq/p_label.json','r',encoding='utf-8') as f :
            loads = json.loads(f.read())
            times_list=loads['date']
            times_list = list(map(str, times_list))
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
                        price_list1.append(loads['data'][mytime]['1'][ind])
                        price_list3.append(loads['data'][mytime]['3'][ind])
                        price_list5.append(loads['data'][mytime]['5'][ind])
                        price_list10.append(loads['data'][mytime]['10'][ind])
                        price_list15.append(loads['data'][mytime]['15'][ind])
                        price_list20.append(loads['data'][mytime]['20'][ind])
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

    with open('../data/nasdaq/p_label_relio.json','w+') as f :
        json.dump(dicts,f)



# 获取json格式的文件
def get_json_lists():

    all_stocks=close_price
    day_list = list(close_price.index)
    get_day_list = day_list[20:]
    length=len(get_day_list)
    n=len(get_day_list)
    #获取一段时间的所有的行业数据
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
    with open('../data/nasdaq/p_label.json','w+') as f :
        json.dump(dist,f)

# 获取json格式的文件
get_json_lists()
# 将数据转化为热力图数据
get_reli_datatype()
print('行业RPS数据更新成功')
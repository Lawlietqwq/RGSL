
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
#使用之前先输入token，可以从个人主页上复制出来，
#每次调用数据需要先运行该命令

ts.set_token('ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4')
pro = ts.pro_api()
"""
endtime:当前时间
get_date_bef(now,n):获取当前时间n年前的时间
"""
# 当前时间
now_time = datetime.datetime.utcnow()+datetime.timedelta(hours=8)
print(now_time)
# 范围时间
start_time = datetime.datetime.strptime(str(now_time.date()) + '16:00', '%Y-%m-%d%H:%M')
# 如果时间小于下午四点，那么数据还没更新，所以使用昨天的数据
if now_time<start_time:
    day_target = now_time+datetime.timedelta(days=-1)
else:
    day_target = now_time
endtime=day_target.strftime("%Y%m%d")
# endtime = time.strftime("%Y%m%d", time.localtime())
end_time = str(endtime)
# 获取一年前的时间
# one_year_time = str(get_date_bef(end_time, 1))

#获取行业代码与名称
df = pro.index_classify(level='L2', src='SW2021',fields='index_code,industry_name')
df.index=df['index_code']


#获取一段时间的所有的行业数据
def getdatas():
    length=len(df)
    datas={}
    times=[]
    count=1
    tscode=list(df['index_code'])
    # for i,tscode in enumerate(df['index_code']):
    while tscode:
        print("代码{}，这是第{}个，还剩{}个".format(tscode[0],count,length-count))
        try:
            #我这里的周期是一年，one_year_time 是选中时间一年前的日期
            df1 = pro.sw_daily(ts_code=tscode[0], start_date="20211102", end_date="20220812", fields='ts_code,trade_date,close')[0:134]
            print("len:",len(df1))
            if len(df1)<134:
                print('{}指数不存在或者数据量太小'.format(tscode[0]))
                tscode.pop(0)
                continue
            datas[tscode[0]]=df1['close']
            #如果提交成功，则将第一个代码剔除，如果失败进入异常，之后循环将会重新提交
            tscode.pop(0)
            count=count+1

            #获取时间序列作为x轴
            if len(times)==0:
                times=df1['trade_date']
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
        s_rank = (all_stocks.iloc[0]/all_stocks.iloc[n]).sort_values(ascending=False).rank(method='dense')
        # 归一
        s_rps = (100*(s_rank - s_rank.min()))/(s_rank.max()-s_rank.min())
        # 写入字典
        s_rps=s_rps.sort_index()
        dict_rps[n] = s_rps.to_dict()
    # print(dict_rps)
    # rps = dict_rps[10]
    # rps[rps > 99]
    return dict_rps

# 将数据转化为热力图数据
def get_reli_datatype():
    industrys_list=[]
    names_list=[]
    times_list=[]
    dicts1={}
    dicts3={}
    dicts5={}
    dicts10={}
    dicts15={}
    dicts20={}
    dicts={}
    with open('p_label.json','r',encoding='utf-8') as f :
            loads = json.loads(f.read())
            times_list=loads.keys()
            for key in loads['20220401']['20'].keys():
                    industrys_list.append(key)
            for ind in industrys_list:
                name=df.at[ind,'industry_name']
                names_list.append(name)
            # for name in names_list:
            for ind in industrys_list:
                    name=df.at[ind,'industry_name']
                    price_list1=[]
                    price_list3=[]
                    price_list5=[]
                    price_list10=[]
                    price_list15=[]
                    price_list20=[]

                    for mytime in times_list:
                            price_list1.append(math.floor(loads[mytime]['1'][ind]))
                            price_list3.append(math.floor(loads[mytime]['3'][ind]))
                            price_list5.append(math.floor(loads[mytime]['5'][ind]))
                            price_list10.append(math.floor(loads[mytime]['10'][ind]))
                            price_list15.append(math.floor(loads[mytime]['15'][ind]))
                            price_list20.append(math.floor(loads[mytime]['20'][ind]))
                            # print(loads[mytime]['50'][ind])
                    # print(price_list10)
                    times_list=list(times_list)
                    dicts1['time']=times_list
                    dicts3['time']=times_list
                    dicts5['time']=times_list
                    dicts10['time']=times_list
                    dicts15['time']=times_list
                    dicts20['time']=times_list
                    dicts1['name']=names_list
                    dicts3['name']=names_list
                    dicts5['name']=names_list
                    dicts10['name']=names_list
                    dicts15['name']=names_list
                    dicts20['name']=names_list
                    dicts1[name]=price_list1
                    dicts3[name]=price_list3
                    dicts5[name]=price_list5
                    dicts10[name]=price_list10
                    dicts15[name]=price_list15
                    dicts20[name]=price_list20
            dicts[1]=dicts1
            dicts[3]=dicts3
            dicts[5]=dicts5
            dicts[10]=dicts10
            dicts[15]=dicts15
            dicts[20]=dicts20

    with open('p_label_reli.json','w+') as f :
        json.dump(dicts,f)

# 获取json格式的文件
def get_json_lists():
    # 获取交易日列表，获取需要更新的日期列表
    # 因为我用的申万数据是新的，数据最新的时间是20210101，而且为什么必须使用[50:]，为了使RPS50可以用上
    get_day_list = pro.trade_cal(start_date='20211102', end_date='20220812')
    get_day_list = list(get_day_list[get_day_list['is_open'] == 1]['cal_date'])[-110:]
    length=len(get_day_list)
    # print(get_day_list)
    n=len(get_day_list)
    #获取一段时间的所有的行业数据
    all_stocks=getdatas()
    print(all_stocks)

    #计算价格
    stocks_price = all_stocks/all_stocks.shift(-1)-1
    stocks_price = stocks_price.fillna(0)

    stocks_price.to_csv('price.csv')
    print(stocks_price)
    #计算连续一段时间的所有行业RPS值
    dict_rpss={}
    while n:
        endday=get_day_list[n-1]
        print("时间:{}这是第{}个，还剩{}个".format(endday,length-n+1,n))
        dict_rps=getRPS(endday,all_stocks)
        dict_rpss[endday]=dict_rps
        n=n-1
    # print(dict_rpss)
    with open('p_label.json','w+') as f :
        json.dump(dict_rpss,f)


# 获取json格式的文件
get_json_lists()
# 将数据转化为热力图数据
get_reli_datatype()
print('行业RPS数据更新成功')
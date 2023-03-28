import tushare as ts
import json
TUSHARE_TOKEN = 'ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# sw_l1 = pro.index_classify(level='L1', src='SW2021')
sw_l2 = pro.index_classify(level='L1', src='SW2021')
sw_l2=sw_l2.groupby(['parent_code'], as_index=False)
print(sw_l2)
from jqdatasdk import *
auth('18974988801', 'Bigdata12345678')
industry = get_industries(name='sw_l1', date='2020-07-07')


industry_dict={}
for sw in sw_l2:
    # print(type(sw[1]))
    industry_list=[]
    for index,s in sw[1].iterrows():
        industry_list.append(s['index_code'])
    industry_dict[sw[0]]=industry_list
with open('../data/rpsdata/industry_relation.json','w+') as f :
    json.dump(industry_dict,f)
# print(industry_dict)
        # print(s['index_code'])
# for index,sw in sw_l1.iterrows():
#     print(sw['industry_code']) s

# print(df_l2.iloc[''])
# print(df_l2.loc[:,'color'])
    # print(sw)

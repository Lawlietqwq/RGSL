

# from jqdatasdk import *
# auth('18974988801', 'Bigdata12345678')
#
# # industry2 = get_industries(name='sw_l2', date="2020-05-07")
# industry = get_industries(name='sw_l1', date='2020-07-07')
# code_lists = list(industry.index)
# import csv
#
# df_index = get_industry_stocks(code_lists[0], date=None)
# industry_df = get_price(df_index[0],start_date='2005-01-04',end_date='2012-01-01',frequency='daily',fields=['close',
#         'high_limit','paused'])
# mm = industry_df['000519.XSHE']
# # with open('../data/rpsdata/code_list.csv', 'w', newline='') as f:
# #     writer = csv.writer(f)
# #     for code in code_lists:
# #         writer.writerow([code])
# for i in code_lists:
#     df1=finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code==i,finance.SW1_DAILY_PRICE.date>='2020-08-20',finance.SW1_DAILY_PRICE.date<='2021-12-10'))
#     print(len(df1))
# # df1=finance.run_query(query(finance.SW1_DAILY_PRICE).filter(finance.SW1_DAILY_PRICE.code=='801024'))
# # print(df1)
# # print('______')
#
# # import tushare as ts
# #
# # TUSHARE_TOKEN = 'ff64b8b56c9ae9eac1389d7827b79dd578a060338fbaa7794fd9c6d4'
# # ts.set_token(TUSHARE_TOKEN)
# # pro = ts.pro_api()
# #
# # industry = pro.index_classify(level='L1')
# # df1 = pro.sw_daily(ts_code='801020.SI')
# # df2 = pro.sw_daily(ts_code='801710.SI',start_date='20150101')
# # df3 = pro.sw_daily(ts_code='801170.SI')
# # print("___________")
# "801740", "801020", "801110", "801160", "801770", "801010", "801120", "801750", "801050", "801890", "801170", "801710", "801780", "801130", "801180", "801760", "801200", "801230", "801880", "801140", "801720", "801080", "801790", "801030", "801210", "801150", "801730", "801040"
x = []
if x:
    print(x)
else:
    print(1)
import pandas as pd
import numpy as np
import tushare as ts
# from matplotlib import pyplot as plt
def calc_ic(df_factor,df_ret,method='spearman'):
    """
    计算ic
    df_factor：因子矩阵(T *N)
    df_ret：价格序列(T *N)
    method：ic的计算方法：'spearman', 'pearson'
    """
    ic = df_factor.corrwith(df_ret,axis=1,method=method)
    # 计算全样本的相关统计数据
    ic_mean,ic_std = ic.mean(),ic.std()
    icir = ic_mean/ic_std
    ic_winrate = len(ic[ic*np.sign(ic_mean)>0])/len(ic[~pd.isna(ic)])
    ic_all = pd.DataFrame([ic_mean,ic_std,icir,ic_winrate],columns=['all'],index=['ic_mean','ic_std','icir','ic_winrate'])
    # ic_by_year = pd.concat([ic_by_year,ic_all.T],axis=0)
    return ic_all,ic



fct_ic = pd.DataFrame()
fct_list = ['up_limit', 'turnover', 'pe', 'pb']
# 涨停数、换手率、pe、pb
up_limit = pd.read_csv('../util/up_limit_dict.csv', index_col=0)
turnover = pd.read_csv('../util/turnover.csv', index_col=0)
pe = pd.read_csv('../util/pe.csv', index_col=0)
pb = pd.read_csv('../util/pb.csv', index_col=0)
#获取RPS
# up_limit.set_index()
rps1 = np.load('../data/rpsdata/rps_data.npy')[:,:,0]
rps1 = pd.DataFrame(rps1, index=pb.index, columns=pb.columns)
ic_all, ic = calc_ic(up_limit,rps1)
fct_ic['up_limit'] = ic_all
ic_all,ic = calc_ic(turnover,rps1)
fct_ic['turnover'] = ic_all
ic_all,ic = calc_ic(pe,rps1)
fct_ic['pe'] = ic_all
ic_all,ic = calc_ic(pb,rps1)
fct_ic['pb'] = ic_all
fct_ic.to_csv('fct_ic.py')
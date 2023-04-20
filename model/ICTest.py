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
    # ic_all = pd.DataFrame([ic_mean,ic_std,icir,ic_winrate],columns=['all'],index=['ic_mean','ic_std','icir','ic_winrate'])
    ic_all = pd.DataFrame([ic_mean,icir,ic_winrate],columns=['all'],index=['IC','IR','IC Winrate'])
    # ic_by_year = pd.concat([ic_by_year,ic_all.T],axis=0)
    return ic_all,ic


def ic_test_ashare():
    fct_ic = pd.DataFrame()
    fct_list = ['up_limit', 'turnover', 'pe', 'pb']
    # 涨停数、换手率、pe、pb
    up_limit = pd.read_csv('../util/up_limit_dict.csv', index_col=0)
    turnover = pd.read_csv('../util/turn_over.csv', index_col=0)
    pct_change = pd.read_csv('../util/pct_change.csv', index_col=0)
    close_price = pd.read_csv('../util/close_price.csv', index_col=0)
    open_price = pd.read_csv('../util/open_price.csv', index_col=0)
    high_price = pd.read_csv('../util/high_price.csv', index_col=0)
    low_price = pd.read_csv('../util/low_price.csv', index_col=0)
    volume = pd.read_csv('../util/volume.csv', index_col=0)
    # pe = pd.read_csv('../util/pe.csv', index_col=0)
    # pb = pd.read_csv('../util/pb.csv', index_col=0)
    #获取RPS
    # up_limit.set_index()
    rps1 = np.load('../data/rpsdata/rps_data.npy')[:,:,0]
    rps1 = pd.DataFrame(rps1, index=up_limit.index, columns=up_limit.columns)
    ic_all, ic = calc_ic(up_limit,rps1)
    fct_ic['up_limit'] = ic_all
    ic_all,ic = calc_ic(turnover,rps1)
    fct_ic['turnover'] = ic_all
    # ic_all,ic = calc_ic(pct_change,rps1)
    # fct_ic['change percent'] = ic_all
    ic_all,ic = calc_ic(close_price,rps1)
    fct_ic['close'] = ic_all
    ic_all,ic = calc_ic(open_price,rps1)
    fct_ic['open'] = ic_all
    ic_all,ic = calc_ic(high_price,rps1)
    fct_ic['high'] = ic_all
    ic_all,ic = calc_ic(low_price,rps1)
    fct_ic['low'] = ic_all
    ic_all,ic = calc_ic(volume,rps1)
    fct_ic['volume'] = ic_all
    fct_ic.to_csv('fct_ic.csv')

def ic_test_nasdaq():
    fct_ic = pd.DataFrame()
    fct_list = ['up_limit', 'turnover', 'pe', 'pb']
    # 涨停数、换手率、pe、pb
    up_limit = pd.read_csv('../data/nasdaq/up_limit_dict.csv', index_col=0)
    turnover = pd.read_csv('../data/nasdaq/turnover.csv', index_col=0).iloc[20:]
    pct_change = pd.read_csv('../data/nasdaq/pct_change.csv', index_col=0).iloc[20:]
    close_price = pd.read_csv('../data/nasdaq/close_price.csv', index_col=0).iloc[20:]
    open_price = pd.read_csv('../data/nasdaq/open_price.csv', index_col=0).iloc[20:]
    high_price = pd.read_csv('../data/nasdaq/high_price.csv', index_col=0).iloc[20:]
    low_price = pd.read_csv('../data/nasdaq/low_price.csv', index_col=0).iloc[20:]
    volume = pd.read_csv('../data/nasdaq/volume.csv', index_col=0).iloc[20:]
    # pe = pd.read_csv('../util/pe.csv', index_col=0).iloc[20:]
    # pb = pd.read_csv('../util/pb.csv', index_col=0).iloc[20:]
    # 获取RPS
    # up_limit.set_index()
    rps1 = np.load('../data/nasdaq/rps_data.npy')[:, :, 0]
    rps1 = pd.DataFrame(rps1, index=up_limit.index, columns=up_limit.columns)
    ic_all, ic = calc_ic(up_limit, rps1)
    fct_ic['up_limit'] = ic_all
    ic_all, ic = calc_ic(turnover, rps1)
    fct_ic['turnover'] = ic_all
    # ic_all,ic = calc_ic(pct_change,rps1)
    # fct_ic['change percent'] = ic_all
    ic_all, ic = calc_ic(close_price, rps1)
    fct_ic['close'] = ic_all
    ic_all, ic = calc_ic(open_price, rps1)
    fct_ic['open'] = ic_all
    ic_all, ic = calc_ic(high_price, rps1)
    fct_ic['high'] = ic_all
    ic_all, ic = calc_ic(low_price, rps1)
    fct_ic['low'] = ic_all
    ic_all, ic = calc_ic(volume, rps1)
    fct_ic['volume'] = ic_all
    fct_ic.to_csv('../data/nasdaq/fct_ic_nasdaq.csv')

ic_test_nasdaq()
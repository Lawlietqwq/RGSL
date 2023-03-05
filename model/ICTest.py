import pandas as pd
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
    # 计算分年度的相关统计数据
    ic_mean_ = ic.groupby(ic.index.year).mean()
    ic_std_ = ic.groupby(ic.index.year).std()
    icir_ = ic_mean_/ic_std_
    ic_winrate_ = ic.groupby(ic.index.year).apply(lambda x:(x*np.sign(ic_mean)>0).sum()/len(x))
    ic_by_year = pd.concat([ic_mean_,ic_std_,icir_,ic_winrate_],axis=1)
    ic_by_year.columns=['ic_mean','ic_std','icir','ic_winrate']
    # 返回分年度和总计的表现
    ic_all = pd.DataFrame([ic_mean,ic_std,icir,ic_winrate],columns=['all'],index=['ic_mean','ic_std','icir','ic_winrate'])
    ic_by_year = pd.concat([ic_by_year,ic_all.T],axis=0)
    return ic_by_year,ic
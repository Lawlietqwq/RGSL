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


df = ak.stock_us_spot_em()
df['原代码'] = df['代码'].replace('_', '')
df['原代码'] = df['原代码'].apply(lambda x:x[4:]).tolist()
code = list(df['原代码'])
symbol = list(df['代码'])
data = {'symbol':symbol, 'code':code}
df = pd.DataFrame(data)
df1 = df.set_index(['code'])['symbol'].to_dict()
with open('../data/nasdaq/dongcai_list.json', 'w+') as f:
    json.dump(df1, f)
print('_____________')

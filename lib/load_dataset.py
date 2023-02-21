import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'rpsdata':
        data_path = os.path.join('../data/rpsdata/rps_data.npy')
        data = np.load(data_path)
        # df = pro.index_classify(level='L2', src='SW2021', fields='index_code,industry_name')
        # df.index = df['index_code']
        # df = pro.index_member(index_code='850531.SI')["out_date"==None].con_code
        # pro.sw_daily()
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

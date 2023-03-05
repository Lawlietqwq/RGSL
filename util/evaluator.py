import math
import numpy as np
# import scipy.stats as sps
# from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import pandas as pd
# 计算MRR和IRR
def evaluate(prediction,ground_truth, report=False):

    # assert ground_truth.shape == prediction.shape, 'shape mis-match'
    price_change = pd.read_csv('pct_change.csv')
    performance = {}
    # performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2\
    #     / np.sum(mask)
    # performance['MAE']=mean_absolute_error(prediction,ground_truth)
    # performance['MSE']=mean_squared_error(prediction,ground_truth)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0

    # print("prediction.shape[1]:",prediction.shape)
    # print("price_change:",price_change)
    # print("ground_truth:",ground_truth)

    return_list=[0.0]
    return_list5=[0.0]
    return_list10=[0.0]
    # IRR_list=[]
    MRR_list=[]
    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(price_change[:, i])
        # print("rank_gt:",(rank_gt))
        # print("ground_truth:",(ground_truth[:, i]))
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])
        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)


        # for j in range(1, prediction.shape[0] + 1):
        # index=prediction.shape[0]
        # gt=ground_truth[:, i]
        # print("gt:",gt)
        # print("rank_gt:",rank_gt)
        for index,cur_rank in enumerate(pre_top1):
            # index-=num
            # cur_rank = rank_pre[-1 * index]
            rel_rank=list(rank_gt).index(cur_rank)+1
            # rank_gt[cur_rank]
            rel_pos=prediction.shape[0]-rel_rank+1
            mrr_top+=1/rel_pos
        MRR_list.append(1/rel_pos)
            # print("预测值第{}的位置：{},真实排名中的名次:{},真实的rps:{},真实的收益率:{}".format(index+1,cur_rank,
            # rel_pos,ground_truth[cur_rank][i],price_change[cur_rank][i]))
        #因为我选用的是前五名的排名，所以最后要除5
        # mrr_top/=5
        # back testing on top 1
        real_ret_rat_top = price_change[list(pre_top1)[0]][i]
        # bt_long *= (1+real_ret_rat_top)
        bt_long += real_ret_rat_top

        real_ret_rat_top = 0
        # print("pre_top5:",pre_top5)
        for pre in pre_top1:
            real_ret_rat_top += price_change[pre][i]
        real_ret_rat_top /= 1
        bt_long = bt_long*(1+real_ret_rat_top)
        return_list.append(bt_long-1)
        # back testing on top 5
        real_ret_rat_top5 = 0
        # print("pre_top5:",pre_top5)
        for pre in pre_top5:
            real_ret_rat_top5 += price_change[pre][i]
            # print("price_change[pre][i]:",price_change[pre][i])
        real_ret_rat_top5 /= 5
        bt_long5 = bt_long5*(1+real_ret_rat_top5)
        return_list5.append(bt_long5-1)
        # back testing on top 10
        real_ret_rat_top10 = 0
        # print("*"*20)
        for pre in pre_top10:
            real_ret_rat_top10 += price_change[pre][i]


            # print("price_change:",price_change[pre][i])
        # print("price_change:",price_change[pre][i])
        real_ret_rat_top10 /= 10
        bt_long10 = bt_long10*(1+real_ret_rat_top10)
        return_list10.append(bt_long10-1)
        # bt_long10 *= (1+real_ret_rat_top10)
        # bt_long10 += real_ret_rat_top10
    print("prediction.shape[1]:",prediction.shape[1])
    performance['MRR'] = mrr_top / (prediction.shape[1])
    # performance['MRR_list'] = MRR_list
    # performance['return_list1'] = return_list
    # performance['return_list5'] = return_list5
    # performance['return_list10'] = return_list10
    # performance['IRR'] = bt_long
    performance['IRR1'] = bt_long-1
    performance['IRR5'] = bt_long5-1
    performance['IRR10'] = bt_long10-1
    # with open('re_TSN_GRU_list.json','w+') as f:
    #     json.dump(performance,f)
    # performance['IRR10'] = bt_long10
    return performance
import csv
import os
import sys

import pandas as pd

from util.section_industry import SectorPreprocessor

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.RGSL import RGSL as Network
from model.com_lstm import LSTM as LSTMNetwork
from model.com_gru import GRU as GRUNetwork
from model.com_rnn import RNN as RNNNetwork
from model.RGSL import RGSL as Network
from model.TestModel import TestModel
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.utils import get_adjacency_matrix, scaled_Laplacian, cheb_polynomial


#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'rpsdata'      #PEMSD4 or PEMSD8
# DATASET = 'PEMSD8'      #PEMSD4 or PEMSD8
DEVICE = 'cuda:0'
# MODEL = 'RGSL'
MODEL = 'AGCRN'

#get configuration
# config_file = './{}_{}.conf'.format(DATASET, MODEL)
config_file = 'test.conf'
print(config_file)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)
print(config.sections())

from lib.metrics import MAE_torch, MSE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def rank_mae_loss(args):
    def loss(preds, labels):
        mse = MSE_torch(pred=preds, true=labels)
        # return_ratio = tf.div(tf.subtract(prediction, base_price), base_price)
        batch_size = len(preds)
        preds_tmp = preds.squeeze(1)
        labels_tmp = labels.squeeze(1)
        all_one = torch.ones(batch_size,28,1).to(args.device)
        pre = torch.sub(
            torch.matmul(preds_tmp, all_one.transpose(1, 2)),
            torch.matmul(all_one, preds_tmp.transpose(1, 2))
        )
        gt = torch.sub(
            torch.matmul(all_one, labels_tmp.transpose(1, 2)),
            torch.matmul(labels_tmp, all_one.transpose(1, 2))
        )
        rank_loss = torch.mean(
            F.relu(
                    torch.multiply(pre, gt),
            )
        )
        res = mse + args.alpha * rank_loss
        return res
    return loss

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--tra_ratio', default=config['data']['tra_ratio'], type=float)
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
args.add_argument('--adj_filename', default=config['data']['adj_filename'], type=str)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#train
args.add_argument('--times', default=config['train']['times'], type=int)
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
args.add_argument('--alpha', default=config['train']['alpha'], type=float, help = 'weight')
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--model-ema-decay', type=float, default=0.999,
                    help='decay factor for model weights moving average (default: 0.9998)')
args = args.parse_args()
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'
class Runner(object):
    def __init__(self):
        #load graph
        if config.has_option('data', 'id_filename'):
            id_filename = config['data']['id_filename']
        else:
            id_filename = None
        # adj_mx, distance_mx = get_adjacency_matrix(args.adj_filename, args.num_nodes, id_filename)
        processor = SectorPreprocessor("", "")

        adj_mx = processor.generate_sector_relation(
                os.path.join('../data/rpsdata/industry_relation.json'),
                'data/rpsdata/code_list.csv'
            )
        adj_mx = adj_mx.astype(np.float32)

        L_tilde = scaled_Laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in cheb_polynomial(L_tilde, args.cheb_k)]

        #init model
        adj_mx = torch.from_numpy(adj_mx).type(torch.FloatTensor).to(args.device)
        L_tilde = torch.from_numpy(L_tilde).type(torch.FloatTensor).to(args.device)
        # model = Network(args, cheb_polynomials, L_tilde)
        # model = LSTMNetwork(args)
        model = TestModel(args)
        # model = RNNNetwork(args)
        # model = GRUNetwork(args)
        # model = RNNNetwork(args.input_dim, args.rnn_units, args.num_layers, args.output_dim,args.batch_size)
        # model = GRUNetwork(args.input_dim, args.rnn_units, args.num_layers, args.output_dim,args.batch_size)

        model = model.to(args.device)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        print_model_parameters(model, only_num=False)

        #load dataset
        train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                                       normalizer=args.normalizer,
                                                                       tod=args.tod, dow=False,
                                                                       weather=False, single=True)

        #init loss function, optimizer
        if args.loss_func == 'mask_mae':
            loss = masked_mae_loss(scaler, mask_value=0.0)
        elif args.loss_func == 'mae':
            loss = torch.nn.SmoothL1Loss().to(args.device)
            # loss = torch.nn.L1Loss().to(args.device)
        elif args.loss_func == 'mse':
            loss = torch.nn.MSELoss().to(args.device)
        elif args.loss_func == 'rank':
            loss = rank_mae_loss(args)
        else:
            raise ValueError

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                     weight_decay=0, amsgrad=False)
        #learning rate decay
        lr_scheduler = None
        if args.lr_decay:
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                                milestones=lr_decay_steps,
                                                                gamma=args.lr_decay_rate)
            #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

        #config log path
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
        args.log_dir = log_dir

        #start training
        self.trainer = Trainer(adj_mx, L_tilde, model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler=lr_scheduler)

result = pd.DataFrame()

if args.mode == 'train':
    new = True
    seed_fields = ['seed', 'epoch', 'lr', 'MAE', 'MRR', 'IRR1', 'IRR5', 'IRR10']
    fields = ['epoch', 'lr', 'MAE', 'MRR', 'IRR1', 'IRR5', 'IRR10']
    avg_fields = ['epoch', 'lr', 'avg_MAE', 'var_MAE', 'avg_MRR', 'var_MRR', 'avg_IRR1', 'var_IRR1', 'avg_IRR5', 'var_IRR5', 'avg_IRR10', 'var_IRR10']
    seed_start = False
    row = {}
    if seed_start:
        # 随机种子
        for seed in range(args.seed):
            init_seed(seed)
            runner = Runner()
            result = runner.trainer.train()
            row['seed'] = seed
            row['epoch'] = args.epochs
            row['lr'] = args.lr_init
            row['MAE'] = result['MAE']
            row['MRR'] = result['MRR']
            row['IRR1'] = result['IRR1']
            row['IRR5'] = result['IRR5']
            row['IRR10'] = result['IRR10']
            with open('../data/performence/log.csv', 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=seed_fields)
                writer.writerow(row)
                f.close()
    else:
        # 平均
        df = pd.DataFrame(columns=fields)
        for i in range(1):
            runner = Runner()
            result = runner.trainer.train()
            row['epoch'] = args.epochs
            row['lr'] = args.lr_init
            row['MAE'] = result['MAE']
            row['MRR'] = result['MRR']
            row['IRR1'] = result['IRR1']
            row['IRR5'] = result['IRR5']
            row['IRR10'] = result['IRR10']
            df = df.append(row,ignore_index=True)
        df.to_csv('../data/performence/avg_run.csv', mode="a")
        row = {}
        row['epoch'] = args.epochs
        row['lr'] = args.lr_init
        row['avg_MAE'] = df['MAE'].mean()
        row['var_MAE'] = df['MAE'].var()
        row['avg_MRR'] = df['MRR'].mean()
        row['var_MRR'] = df['MRR'].var()
        row['avg_IRR1'] = df['IRR1'].mean()
        row['var_IRR1'] = df['IRR1'].var()
        row['avg_IRR5'] = df['IRR5'].mean()
        row['var_IRR5'] = df['IRR5'].var()
        row['avg_IRR10'] = df['IRR10'].mean()
        row['var_IRR10'] = df['IRR10'].var()
        with open(file='../data/performence/avg_log.csv', mode = "a") as f:
            writer = csv.DictWriter(f, fieldnames=avg_fields)
            if new:
                writer.writeheader()
            writer.writerow(row)

# elif args.mode == 'test':
#     output = {}
#     model.load_state_dict(torch.load('./best_model.pth'.format(args.dataset)))
#     x = model.node_embeddings
#     L_tilde_learned = F.relu(torch.mm(x, x.transpose(0, 1))).cpu().detach().numpy()
#     node_embedding = model.node_embeddings.cpu().detach().numpy()
#
#     L_tilde = L_tilde.cpu().detach().numpy()
#     adj_mx = adj_mx.cpu().detach().numpy()
#
#     print("Load saved model")
#     trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
#     adj_learned = model.adj.cpu().detach().numpy()
#     tilde_learned = model.tilde.cpu().detach().numpy()
#
#     np.savetxt("node_embedding.txt", node_embedding, fmt="%s",delimiter=",")
#     np.savetxt("L_tilde_learned.txt", tilde_learned, fmt="%s",delimiter=",")
#     np.savetxt("L_tilde.txt", L_tilde, fmt="%s",delimiter=",")
#     np.savetxt("adj_mx.txt", adj_mx, fmt="%s",delimiter=",")
#     np.savetxt("adj_learned.txt", adj_learned, fmt="%s",delimiter=",")
# else:
#     raise ValueError

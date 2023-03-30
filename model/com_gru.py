import torch
import torch.nn as nn
from util.evaluator import evaluate


# 定义GRU网络
class GRU(nn.Module):
    # def __init__(self, feature_size, hidden_size, num_layers, output_size, batch_size):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.hidden_size = args.rnn_units  # 隐层大小
        self.num_layers = args.num_layers  # gru层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为7
        self.gru = nn.GRU(args.input_dim*args.num_nodes, args.rnn_units, args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.rnn_units, args.output_dim*args.num_nodes)
        self.batch_size = args.batch_size
        self.args = args
        self.node_num = args.num_nodes

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小
        seq_length = x.shape[1]

        x = x.reshape(batch_size, seq_length, -1)
        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden


        # GRU运算
        output, h_0 = self.gru(x, h_0)

        # 获取GRU输出的维度信息
        batch_size, timestep, hidden_size = output.shape

        # 将output变成 batch_size * timestep, hidden_dim
        output = output.reshape(-1, hidden_size)

        # 全连接层
        output = self.fc(output)  # 形状为batch_size * timestep, 1

        # 转换维度，用于输出
        output = output.reshape(batch_size, timestep, -1, 1)

        # 我们只需要返回最后一个时间片的数据即可
        return output[:, -1:, ...]

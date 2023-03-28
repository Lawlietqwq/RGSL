import torch
import torch.nn as nn
from util.evaluator import evaluate


class TestModel(nn.Module):
    def __init__(self, args):
        super(TestModel, self).__init__()
        self.batch_size = args.batch_size
        self.args = args
        self.node_num = args.num_nodes
        self.seq_length = args.lag
        self.input_dim = args.input_dim
        self.end_conv = nn.Conv1d(self.seq_length*self.input_dim, args.horizon * args.output_dim, kernel_size=1, bias=True)

    def forward(self, x, hidden=None):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(self.batch_size, self.node_num, self.input_dim*self.seq_length)
        x = x.permute(0, 2, 1)
        output = self.end_conv(x)
        output = output.unsqueeze(-1)
        return output


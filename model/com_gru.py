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
        self.gru = nn.GRU(args.input_dim, args.rnn_units, args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.rnn_units, args.output_dim)
        self.batch_size = args.batch_size
        self.args = args
        self.node_num = args.num_nodes
        init_states = []
        for i in range(self.num_layers):
            init_states.append(torch.zeros(self.batch_size, self.node_num, self.hidden_dim))
        self.init_states = torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小
        seq_length = x.shape[1]
        # 初始化隐层状态
        # if hidden is None:
        #     h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        # else:
        #     h_0 = hidden
        output_hidden = []
        current_inputs = x
        for i in range(self.num_layers):
            state = self.init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.GRUCell[i](current_inputs[:, t, :, :], state)
                state = state.to(x.device)
                input_and_state = torch.cat((x, state), dim=-1)
                z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, learned_tilde))
                z, r = torch.split(z_r, self.hidden_dim, dim=-1)
                candidate = torch.cat((x, z * state), dim=-1)
                hc = torch.tanh(self.update(candidate, node_embeddings, learned_tilde))
                state = r * state + (1 - r) * hc
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

        # # GRU运算
        # output, h_0 = self.gru(x, h_0)
        #
        # # 获取GRU输出的维度信息
        # batch_size, timestep, hidden_size = output.shape
        #
        # # 将output变成 batch_size * timestep, hidden_dim
        # output = output.reshape(-1, hidden_size)
        #
        # # 全连接层
        # output = self.fc(output)  # 形状为batch_size * timestep, 1
        #
        # # 转换维度，用于输出
        # output = output.reshape(timestep, batch_size, -1)
        #
        # # 我们只需要返回最后一个时间片的数据即可
        # return output[-1]

class GRUCell(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(RGSLCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out

    def forward(self, x, h0, w_ih, w_hh, b_ih, b_hh):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        bs, T, h_in = x.shape
        h_out = w_ih.shape[0] // 3
        # w_ih.shape=torch.Size([3*h_out,h_in])
        # batch_w_ih.shape = torch.Size([bs,3*h_out,h_in])
        batch_w_ih = w_ih.unsqueeze(0).tile([bs, 1, 1])
        # h0.shape=prev_h.shape=torch.Size([bs,h_out])
        prev_h = h0

        # w_hh.shape=torch.Size([3*h_out,h_out])
        # batch_w_hh=torch.Size([bs,3*h_out,h_out])
        batch_w_hh = w_hh.unsqueeze(0).tile([bs, 1, 1])
        output = torch.zeros([bs, T, h_out])

        # input.shape=torch.Size([bs,T,h_in])
        # x.shape=torch.Size([bs,h_in])->([bs,h_in,1])
        x = x[:, t, :].unsqueeze(-1)
        # batch_w_ih.shape=torch.Size([bs,3*h_out,h_in])
        # w_ih_times_x.shape=torch.Size([bs,3*h_out,1])->([bs,3*h_out])
        w_ih_times_x = torch.bmm(batch_w_ih, x).squeeze(-1)
        # batch_w_hh.shape=torch.Size([bs,3*h_out,h_out])
        # prev_h.shape=torch.Size([bs,h_out])->([bs,h_out,1])
        # w_hh_times_x.shape=torch.Size([bs,3*h_out,1]) ->([bs,3*h_out])
        w_hh_times_x = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1)).squeeze(-1)

        r_t = torch.sigmoid(w_ih_times_x[:, :h_out] + b_ih[:h_out] + w_hh_times_x[:, :h_out] + b_hh[:h_out])
        z_t = torch.sigmoid(
            w_ih_times_x[:, h_out:2 * h_out] + b_ih[h_out:2 * h_out] + w_hh_times_x[:, h_out:2 * h_out] + b_hh[
                                                                                                          h_out:2 * h_out])
        n_t = torch.tanh(w_ih_times_x[:, 2 * h_out:3 * h_out] + b_ih[2 * h_out:3 * h_out] + r_t * (
                    w_hh_times_x[:, 2 * h_out:3 * h_out] + b_hh[2 * h_out:3 * h_out]))
        prev_h = (1 - z_t) * n_t + z_t * prev_h

        output[:, t, :] = prev_h
        # prev_h.shape=torch.Size([bs,h_out])
        # h_n.shape=torch.Size([1,bs,h_out])

        h_n = prev_h.unsqueeze(0)

        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
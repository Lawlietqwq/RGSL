import torch
from torch.autograd import Variable
from torch import nn


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()

        # Defining some parameters
        self.input_size = args.input_dim
        self.hidden_dim = args.rnn_units
        self.num_layers = args.num_layers
        self.output_size = args.output_dim
        self.num_nodes = args.num_nodes
        self.batch_size = args.batch_size
        self.args = args
        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(self.num_nodes*self.input_size, self.hidden_dim, self.num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, self.num_nodes*self.output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = x.reshape(batch_size, seq_len, self.num_nodes*self.input_size)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = out[:, -1, :]
        out = out.unsqueeze(-1).unsqueeze(1)
        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.args.device)
        return hidden
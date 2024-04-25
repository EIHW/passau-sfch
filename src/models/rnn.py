import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from global_vars import device


class GRUClassifier(nn.Module):
    '''
    GRU layers (bidirectional or unidirectional) + Dropout + Linear
    '''

    def __init__(self, num_classes, params):
        super().__init__()
        self.num_layers = params.rnn_num_layers
        self.hidden_dim = params.rnn_hidden_dim
        self.num_directions = 2 if params.bidirectional else 1

        self.gru = nn.GRU(input_size=params.feature_dim, hidden_size=params.rnn_hidden_dim, num_layers=params.rnn_num_layers,
                          batch_first=True, dropout=params.rnn_dropout, bidirectional=params.bidirectional)
        self.dropout = nn.Dropout(params.dropout)
        self.rnn_out_dim = (2 if params.bidirectional else 1) * params.rnn_hidden_dim
        self.linear = nn.Linear(self.rnn_out_dim, 1 if num_classes==2 else num_classes)


    def forward(self, input, lengths):
        x = pack_padded_sequence(input, lengths.cpu(), batch_first=True, enforce_sorted=False) # BS, SL, feature_dim
        rnn_enc = self.gru(x) # rnn_dim, BS,
        h_n = rnn_enc[1]  # (ND*NL, BS, dim)
        batch_size = h_n.shape[1]
        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_dim)  # (NL, ND, BS, dim)
        last_layer = h_n[-1].permute(1, 0, 2)  # (BS, ND, dim)
        x_out = last_layer.reshape(batch_size, self.num_directions * self.hidden_dim) # (BS, ND*dim)

        return self.linear(self.dropout(x_out))


class GRUEncoder(nn.Module):
    '''
    GRU layers (bidirectional or unidirectional)
    '''

    def __init__(self, input_size, hidden_dim):
        super().__init__()
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.num_directions = 2

        self.gru = nn.GRU(input_size=input_size, hidden_size=int(self.hidden_dim/2), num_layers=self.num_layers,
                          batch_first=True, bidirectional=self.num_directions==2)
        self.rnn_out_dim = self.num_directions * self.hidden_dim


    def forward(self, input, mask):
        lengths = torch.sum(mask.detach(), dim=1)
        x = pack_padded_sequence(input, lengths.cpu(), batch_first=True, enforce_sorted=False) # BS, SL, feature_dim
        rnn_enc = self.gru(x)
        outs = rnn_enc[0] # (BS, SL, dim)
        outs, _ = pad_packed_sequence(outs, batch_first=True)
        if mask.shape[1] > torch.max(lengths):
            pad_tensor = torch.zeros((mask.shape[0], mask.shape[1] - int(torch.max(lengths).item()), outs.shape[2]))
            outs = torch.concatenate([outs, pad_tensor.to(device)], dim=1)

        return outs



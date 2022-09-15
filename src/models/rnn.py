import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


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
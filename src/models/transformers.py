from argparse import Namespace

import torch.nn as nn
import torch


class FFN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, input_seq):
        return self.linear2(self.relu(self.linear1(input_seq)))


class CustomTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.ffn = FFN(input_dim, hidden_dim)
        self.do1 = torch.nn.Dropout(dropout)
        self.do2 = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask):
        x = self.mha(query, key, value, key_padding_mask=key_padding_mask)[0]
        x = self.do1(x) + value
        x = self.norm1(x)
        x_ff = self.ffn(x)
        x = x + self.do2(x_ff)
        x = self.norm2(x)
        return x


class CustomMM(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_projection = nn.Linear(params.a_dim, params.v_dim, bias=False)
        self.t_projection = nn.Linear(params.t_dim, params.v_dim, bias=False)
        self.tanh = nn.Tanh()

        self.pos_v = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.v_dim)
        self.pos_a = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.v_dim)
        self.pos_t = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.v_dim)

        v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.v_dim, nhead=params.trf_num_heads,
                                                              batch_first=True)
        self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.trf_num_v_layers)

        a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.v_dim, nhead=params.trf_num_heads,
                                                              batch_first=True)
        self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.trf_num_at_layers)
        t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.v_dim, nhead=params.trf_num_heads,
                                                         batch_first=True)
        self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.trf_num_at_layers)

        self.v2a_transformer = CustomTransformerEncoderLayer(input_dim = params.v_dim, hidden_dim=2048,
                                                             num_heads=params.trf_num_heads, dropout=0.1)
        self.v2t_transformer = CustomTransformerEncoderLayer(input_dim=params.v_dim, hidden_dim=2048,
                                                             num_heads=params.trf_num_heads, dropout=0.1)

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(3*params.v_dim, 1)

        self.pooling = nn.MaxPool2d(kernel_size=(4,1), stride=(2,1))


    def forward(self, v, a, t, mask):
        a = self.tanh(self.a_projection(a)) # BS, SL, dim
        t = self.tanh(self.t_projection(t)) # BS, SL, dim

        # Todo positional embeddings
        emb_indices = indices_from_mask(mask)
        v_pos = self.pos_v(emb_indices)
        a_pos = self.pos_a(emb_indices)
        t_pos = self.pos_t(emb_indices)
        v = v + v_pos
        a = a + a_pos
        t = t + t_pos


        trf_mask = ~mask.bool()
        v = self.v_transformer(v, src_key_padding_mask = trf_mask) # BS, SL, dim
        a = self.a_transformer(a, src_key_padding_mask = trf_mask) # BS, SL ,dim
        t = self.t_transformer(t, src_key_padding_mask = trf_mask) # BS, SL, dim

        a = self.v2a_transformer(query=v, key=a, value=a, key_padding_mask=trf_mask)
        t = self.v2t_transformer(query=v, key=t, value=t, key_padding_mask=trf_mask)

        representation = torch.concatenate([v,a,t], dim=-1)
        representation = self.pooling(representation)
        return self.classification(self.dropout(representation))


def indices_from_mask(mask):
    full_row = torch.range(1, mask.shape[-1])
    full_indices = torch.vstack([full_row]*mask.shape[0])
    masked = full_indices * mask
    return masked.long()

# test
# dim_v = 32
# dim_a = 14
# dim_t = 20
# BS = 2
# SL1 = 5
# SL2 = 7
# max_sl = max(SL1, SL2)
#
# NH = 4
#
# v = torch.randn((BS, max_sl, dim_v))
# a = torch.randn((BS, max_sl, dim_a))
# t = torch.randn((BS, max_sl, dim_t))
#
# mask = torch.ones(BS, max_sl)
# mask[0, SL1:] = 0.
# mask[1, SL2:] = 0.
#
# for i,l in enumerate([SL1, SL2]):
#     v[i, l:] = 0.
#     a[i, l:] = 0.
#     t[i, l:] = 0.
#
# params = Namespace(**{
#     'v_dim': dim_v,
#     'a_dim': dim_a,
#     't_dim': dim_t,
#     'num_heads': NH,
#     'num_at_layers': 1,
#     'num_v_layers': 1,
#     'max_length': max_sl
# })
#
# model = CustomMM(params)
# model.eval()
#
# test_out = model(v, a, t, mask)
# print('Test')


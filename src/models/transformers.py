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
        self.a_projection = nn.Linear(params.a_dim, params.v_dim)
        self.t_projection = nn.Linear(params.t_dim, params.v_dim)
        self.relu = nn.ReLU()

        v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.v_dim, nhead=params.num_heads,
                                                              batch_first=True)
        self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.num_v_layers)

        a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.v_dim, nhead=params.num_heads,
                                                              batch_first=True)
        self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.num_at_layers)
        t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.v_dim, nhead=params.num_heads,
                                                         batch_first=True)
        self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.num_at_layers)

        # TODO replace with custom
        self.v2a_transformer = CustomTransformerEncoderLayer(input_dim = params.v_dim, hidden_dim=2048,
                                                             num_heads=params.num_heads, dropout=0.1)
        self.v2t_transformer = CustomTransformerEncoderLayer(input_dim=params.v_dim, hidden_dim=2048,
                                                             num_heads=params.num_heads, dropout=0.1)

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(3*params.v_dim, 2)


    def forward(self, v, a, t, mask):
        a = self.relu(self.a_projection(a)) # BS, SL, dim
        t = self.relu(self.t_projection(t)) # BS, SL, dim

        v = self.v_transformer(v, key_padding_mask = mask) # BS, SL, dim
        a = self.a_transformer(a, key_padding_mask = mask) # BS, SL ,dim
        t = self.t_transformer(t, key_padding_mask = mask) # BS, SL, dim

        a = self.v2a_transformer(query=v, key=a, value=a, key_padding_mask=mask)
        t = self.v2t_transformer(query=v, key=t, value=t, key_padding_mask=mask)

        representation = torch.concatenate([v,a,t], dim=1)
        return self.classification(self.dropout(representation))


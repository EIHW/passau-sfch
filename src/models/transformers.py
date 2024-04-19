from argparse import Namespace

import torch.nn as nn
import torch
import numpy as np

from global_vars import device

SINUSOID = 'sinus'
LEARNABLE = 'learn'
EMB_TYPES = [SINUSOID, LEARNABLE]


class FFN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, input_seq):
        return self.linear2(self.relu(self.linear1(input_seq)))


# class LRScheduler(LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000, **kwargs):
#         super(LRScheduler, self).__init__(**kwargs)
#
#         self.d_model = cast(d_model, float32)
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step_num):
#
#         # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
#         arg1 = step_num ** -0.5
#         arg2 = step_num * (self.warmup_steps ** -1.5)
#
#         return (self.d_model ** -0.5) * math.minimum(arg1, arg2)


class CustomTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.ffn = FFN(input_dim, hidden_dim)
        self.do1 = torch.nn.Dropout(dropout)
        self.do2 = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask, attn_mask):
        x, att = self.mha(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.do1(x) + value
        x = self.norm1(x)
        x_ff = self.ffn(x)
        x = x + self.do2(x_ff)
        x = self.norm2(x)
        return x


class CustomMM(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_projection = nn.Linear(params.a_dim, params.trf_model_dim)
        self.t_projection = nn.Linear(params.t_dim, params.trf_model_dim)
        self.v_projection = nn.Linear(params.v_dim, params.trf_model_dim)
        self.tanh = nn.Tanh()

        self.num_heads = params.trf_num_heads

        # TODO weight sharing?
        if params.trf_pos_emb == LEARNABLE:
            self.pos_v = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_a = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_t = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
        elif params.trf_pos_emb == SINUSOID:
            self.pos_v = create_positional_embeddings_matrix(max_seq_len=params.max_length, dim=params.trf_model_dim)
            self.pos_a = self.pos_v
            self.pos_t = self.pos_v


        if params.trf_num_v_layers > 0:
            v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
            self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.trf_num_v_layers)
        else:
            self.v_transformer = None

        if params.trf_num_at_layers > 0:
            a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
            self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.trf_num_at_layers)
            t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                         batch_first=True, dim_feedforward=2*params.trf_model_dim)
            self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.trf_num_at_layers)
        else:
            self.a_transformer = None
            self.t_transformer = None

        self.v2a_transformer = CustomTransformerEncoderLayer(input_dim = params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1, hidden_dim=2*params.trf_model_dim)
        self.v2t_transformer = CustomTransformerEncoderLayer(input_dim=params.trf_model_dim, hidden_dim=2*params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1)

        if params.trf_num_mm_layers > 0:
            mm_transformer_layer = nn.TransformerEncoderLayer(d_model=3*params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=6*params.trf_model_dim)
            self.mm_transformer = nn.TransformerEncoder(mm_transformer_layer, num_layers=params.trf_num_mm_layers)
        else:
            self.mm_transformer = None

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(3*params.trf_model_dim, 1)

        self.pooling = nn.MaxPool2d(kernel_size=(4,1), stride=(2,1))


    def forward(self, v:torch.Tensor, a, t, mask):
        #print(v.get_device())
        #print(a.get_device())
        #print(t.get_device())
        #print(self.a_projection.weight.get_device())
        #print(self.a_projection.bias.get_device())
        v = self.tanh(self.v_projection(v))
        a = self.tanh(self.a_projection(a)) # BS, SL, dim
        t = self.tanh(self.t_projection(t)) # BS, SL, dim

        emb_indices = indices_from_mask(mask)
        v_pos = self.pos_v(emb_indices)
        a_pos = self.pos_a(emb_indices)
        t_pos = self.pos_t(emb_indices)
        v = v + v_pos
        a = a + a_pos
        t = t + t_pos


        trf_key_mask = ~mask.bool()
        trf_3d_mask = ~get_3d_mask(mask, self.num_heads).bool()
        if not self.v_transformer is None:
            v = self.v_transformer(v, src_key_padding_mask = trf_key_mask, mask =trf_3d_mask) # BS, SL, dim
        if not self.a_transformer is None:
            a = self.a_transformer(a, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL ,dim
            t = self.t_transformer(t, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL, dim

        a = self.v2a_transformer(query=v, key=a, value=a, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        t = self.v2t_transformer(query=v, key=t, value=t, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)

        representation = torch.concatenate([v,a,t], dim=-1)

        if not self.mm_transformer is None:
            representation = self.mm_transformer(representation, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask)

        representation = self.pooling(representation)
        return self.classification(self.dropout(representation))


def indices_from_mask(mask):
    #print(mask.get_device())
    full_row = torch.range(1, mask.shape[-1]).to(device)
    #print(full_row.get_device())
    full_indices = torch.vstack([full_row]*mask.shape[0])
    #print(full_indices.get_device())
    masked = full_indices * mask
    return masked.long()

def get_3d_mask(mask, num_heads):
    # mask is initially BS, SL
    bs = mask.shape[0]
    sl = mask.shape[1]
    #mask3d = mask.unsqueeze(-1) # BS, SL, 1
    mask3d = torch.repeat_interleave(mask, sl, dim=0) # BS, SL*SL
    mask3d = torch.reshape(mask3d, (bs, sl, sl)) # BS, SL, SL
    mask3d = torch.repeat_interleave(mask3d, num_heads, dim=0) # BS*NH, SL
    #control = torch.sum(mask3d, dim=-1) # just for debugging
    #control = torch.mean(control, dim=-1)
    return mask3d

# from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
def create_positional_embeddings_matrix(max_seq_len, dim, n=10000):
    matrix = np.zeros((max_seq_len+1, dim)) # +1 because of zero padding
    for k in range(max_seq_len):
        for i in np.arange(int(dim/2)):
            denominator = np.power(n, 2*i/dim)
            matrix[k, 2*i] = np.sin(k/denominator)
            matrix[k, 2*i+1] = np.cos(k/denominator)
    return torch.nn.Embedding(num_embeddings=matrix.shape[0], embedding_dim = dim, padding_idx=0, _freeze=True)

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

FULL = 'full'
NO_POS = 'no_pos'
NO_FUSION = 'no_fusion'
V_ONLY = 'v_only'
AVG = 'avg_pool'
AVG_LOGITS = 'avg_logits'

MODEL_TYPES = [FULL, NO_POS, NO_FUSION, V_ONLY, AVG, AVG_LOGITS]

def get_model_class(model_type):
    if model_type == FULL:
        return CustomMM
    elif model_type == NO_POS:
        return CustomMM_NO_POS
    elif model_type == NO_FUSION:
        return CustomMM_NoFusion
    elif model_type == V_ONLY:
        return V_only
    elif model_type == AVG:
        return CustomMM_Avg
    elif model_type == AVG_LOGITS:
        return CustomMM_AvgAfterLogits
    else:
        raise NotImplementedError()

# DUMMY CLASSES
class CustomMM_NO_POS(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_projection = nn.Linear(params.a_dim, params.trf_model_dim)
        self.t_projection = nn.Linear(params.t_dim, params.trf_model_dim)
        self.v_projection = nn.Linear(params.v_dim, params.trf_model_dim)
        self.tanh = nn.Tanh()

        self.num_heads = params.trf_num_heads

        v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.trf_num_v_layers)

        a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.trf_num_at_layers)
        t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                         batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.trf_num_at_layers)

        self.v2a_transformer = CustomTransformerEncoderLayer(input_dim = params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1, hidden_dim=2*params.trf_model_dim)
        self.v2t_transformer = CustomTransformerEncoderLayer(input_dim=params.trf_model_dim, hidden_dim=2*params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1)

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(3*params.trf_model_dim, 1)

        self.pooling = nn.MaxPool2d(kernel_size=(4,1), stride=(2,1))


    def forward(self, v:torch.Tensor, a, t, mask):
        #print(v.get_device())
        #print(a.get_device())
        #print(t.get_device())
        #print(self.a_projection.weight.get_device())
        #print(self.a_projection.bias.get_device())
        v = self.tanh(self.v_projection(v))
        a = self.tanh(self.a_projection(a)) # BS, SL, dim
        t = self.tanh(self.t_projection(t)) # BS, SL, dim

        # emb_indices = indices_from_mask(mask)
        # v_pos = self.pos_v(emb_indices)
        # a_pos = self.pos_a(emb_indices)
        # t_pos = self.pos_t(emb_indices)
        # v = v + v_pos
        # a = a + a_pos
        # t = t + t_pos


        trf_key_mask = ~mask.bool()
        trf_3d_mask = ~get_3d_mask(mask, self.num_heads).bool()
        v = self.v_transformer(v, src_key_padding_mask = trf_key_mask, mask =trf_3d_mask) # BS, SL, dim
        a = self.a_transformer(a, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL ,dim
        t = self.t_transformer(t, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL, dim

        a = self.v2a_transformer(query=v, key=a, value=a, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        t = self.v2t_transformer(query=v, key=t, value=t, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)

        representation = torch.concatenate([v,a,t], dim=-1)
        representation = self.pooling(representation)
        return self.classification(self.dropout(representation))



class CustomMM_NoFusion(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_projection = nn.Linear(params.a_dim, params.trf_model_dim)
        self.t_projection = nn.Linear(params.t_dim, params.trf_model_dim)
        self.v_projection = nn.Linear(params.v_dim, params.trf_model_dim)
        self.tanh = nn.Tanh()

        self.num_heads = params.trf_num_heads

        # TODO weight sharing?
        if params.trf_pos_emb == LEARNABLE:
            self.pos_v = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_a = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_t = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
        elif params.trf_pos_emb == SINUSOID:
            self.pos_v = create_positional_embeddings_matrix(max_seq_len=params.max_length, dim=params.trf_model_dim)
            self.pos_a = self.pos_v
            self.pos_t = self.pos_v


        v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.trf_num_v_layers)

        a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.trf_num_at_layers)
        t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                         batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.trf_num_at_layers)

        # self.v2a_transformer = CustomTransformerEncoderLayer(input_dim = params.trf_model_dim,
        #                                                      num_heads=params.trf_num_heads, dropout=0.1, hidden_dim=2*params.trf_model_dim)
        # self.v2t_transformer = CustomTransformerEncoderLayer(input_dim=params.trf_model_dim, hidden_dim=2*params.trf_model_dim,
        #                                                      num_heads=params.trf_num_heads, dropout=0.1)

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(3*params.trf_model_dim, 1)

        self.pooling = nn.MaxPool2d(kernel_size=(4,1), stride=(2,1))


    def forward(self, v:torch.Tensor, a, t, mask):
        #print(v.get_device())
        #print(a.get_device())
        #print(t.get_device())
        #print(self.a_projection.weight.get_device())
        #print(self.a_projection.bias.get_device())
        v = self.tanh(self.v_projection(v))
        a = self.tanh(self.a_projection(a)) # BS, SL, dim
        t = self.tanh(self.t_projection(t)) # BS, SL, dim

        emb_indices = indices_from_mask(mask)
        v_pos = self.pos_v(emb_indices)
        a_pos = self.pos_a(emb_indices)
        t_pos = self.pos_t(emb_indices)
        v = v + v_pos
        a = a + a_pos
        t = t + t_pos


        trf_key_mask = ~mask.bool()
        trf_3d_mask = ~get_3d_mask(mask, self.num_heads).bool()
        v = self.v_transformer(v, src_key_padding_mask = trf_key_mask, mask =trf_3d_mask) # BS, SL, dim
        a = self.a_transformer(a, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL ,dim
        t = self.t_transformer(t, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL, dim

        #a = self.v2a_transformer(query=v, key=a, value=a, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        #t = self.v2t_transformer(query=v, key=t, value=t, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)

        representation = torch.concatenate([v,a,t], dim=-1)
        representation = self.pooling(representation)
        return self.classification(self.dropout(representation))



class V_only(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.a_projection = nn.Linear(params.a_dim, params.trf_model_dim)
        #self.t_projection = nn.Linear(params.t_dim, params.trf_model_dim)
        self.v_projection = nn.Linear(params.v_dim, params.trf_model_dim)
        self.tanh = nn.Tanh()

        self.num_heads = params.trf_num_heads

        # TODO weight sharing?
        if params.trf_pos_emb == LEARNABLE:
            self.pos_v = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            #self.pos_a = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            #self.pos_t = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
        elif params.trf_pos_emb == SINUSOID:
            self.pos_v = create_positional_embeddings_matrix(max_seq_len=params.max_length, dim=params.trf_model_dim)
            #self.pos_a = self.pos_v
            #self.pos_t = self.pos_v


        v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.trf_num_v_layers)

        #a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
        #                                                      batch_first=True, dim_feedforward=2*params.trf_model_dim)
        #self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.trf_num_at_layers)
        #t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
        #                                                 batch_first=True, dim_feedforward=2*params.trf_model_dim)
        #self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.trf_num_at_layers)

        #self.v2a_transformer = CustomTransformerEncoderLayer(input_dim = params.trf_model_dim,
        #                                                     num_heads=params.trf_num_heads, dropout=0.1, hidden_dim=2*params.trf_model_dim)
        #self.v2t_transformer = CustomTransformerEncoderLayer(input_dim=params.trf_model_dim, hidden_dim=2*params.trf_model_dim,
        #                                                     num_heads=params.trf_num_heads, dropout=0.1)

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(params.trf_model_dim, 1)

        self.pooling = nn.MaxPool2d(kernel_size=(4,1), stride=(2,1))


    def forward(self, v:torch.Tensor, a, t, mask):
        #print(v.get_device())
        #print(a.get_device())
        #print(t.get_device())
        #print(self.a_projection.weight.get_device())
        #print(self.a_projection.bias.get_device())
        v = self.tanh(self.v_projection(v))
        #a = self.tanh(self.a_projection(a)) # BS, SL, dim
        #t = self.tanh(self.t_projection(t)) # BS, SL, dim

        emb_indices = indices_from_mask(mask)
        v_pos = self.pos_v(emb_indices)
        #a_pos = self.pos_a(emb_indices)
        #t_pos = self.pos_t(emb_indices)
        v = v + v_pos
        #a = a + a_pos
        #t = t + t_pos


        trf_key_mask = ~mask.bool()
        trf_3d_mask = ~get_3d_mask(mask, self.num_heads).bool()
        v = self.v_transformer(v, src_key_padding_mask = trf_key_mask, mask =trf_3d_mask) # BS, SL, dim
        #a = self.a_transformer(a, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL ,dim
        #t = self.t_transformer(t, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL, dim

        #a = self.v2a_transformer(query=v, key=a, value=a, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        #t = self.v2t_transformer(query=v, key=t, value=t, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)

        #representation = torch.concatenate([v,a,t], dim=-1)
        representation = self.pooling(v)
        #first_few = v[0, :4, :]
        return self.classification(self.dropout(representation))




class CustomMM_Avg(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_projection = nn.Linear(params.a_dim, params.trf_model_dim)
        self.t_projection = nn.Linear(params.t_dim, params.trf_model_dim)
        self.v_projection = nn.Linear(params.v_dim, params.trf_model_dim)
        self.tanh = nn.Tanh()

        self.num_heads = params.trf_num_heads

        # TODO weight sharing?
        if params.trf_pos_emb == LEARNABLE:
            self.pos_v = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_a = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_t = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
        elif params.trf_pos_emb == SINUSOID:
            self.pos_v = create_positional_embeddings_matrix(max_seq_len=params.max_length, dim=params.trf_model_dim)
            self.pos_a = self.pos_v
            self.pos_t = self.pos_v


        v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.trf_num_v_layers)

        a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.trf_num_at_layers)
        t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                         batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.trf_num_at_layers)

        self.v2a_transformer = CustomTransformerEncoderLayer(input_dim = params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1, hidden_dim=2*params.trf_model_dim)
        self.v2t_transformer = CustomTransformerEncoderLayer(input_dim=params.trf_model_dim, hidden_dim=2*params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1)

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(3*params.trf_model_dim, 1)

        self.pooling = nn.AvgPool2d(kernel_size=(4,1), stride=(2,1))

    def forward(self, v:torch.Tensor, a, t, mask):
        #print(v.get_device())
        #print(a.get_device())
        #print(t.get_device())
        #print(self.a_projection.weight.get_device())
        #print(self.a_projection.bias.get_device())
        v = self.tanh(self.v_projection(v))
        a = self.tanh(self.a_projection(a)) # BS, SL, dim
        t = self.tanh(self.t_projection(t)) # BS, SL, dim

        emb_indices = indices_from_mask(mask)
        v_pos = self.pos_v(emb_indices)
        a_pos = self.pos_a(emb_indices)
        t_pos = self.pos_t(emb_indices)
        v = v + v_pos
        a = a + a_pos
        t = t + t_pos


        trf_key_mask = ~mask.bool()
        trf_3d_mask = ~get_3d_mask(mask, self.num_heads).bool()
        v = self.v_transformer(v, src_key_padding_mask = trf_key_mask, mask =trf_3d_mask) # BS, SL, dim
        a = self.a_transformer(a, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL ,dim
        t = self.t_transformer(t, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL, dim

        a = self.v2a_transformer(query=v, key=a, value=a, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        t = self.v2t_transformer(query=v, key=t, value=t, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)

        representation = torch.concatenate([v,a,t], dim=-1)
        representation = self.pooling(representation)
        return self.classification(self.dropout(representation))


class CustomMM_AvgAfterLogits(nn.Module):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_projection = nn.Linear(params.a_dim, params.trf_model_dim)
        self.t_projection = nn.Linear(params.t_dim, params.trf_model_dim)
        self.v_projection = nn.Linear(params.v_dim, params.trf_model_dim)
        self.tanh = nn.Tanh()

        self.num_heads = params.trf_num_heads

        # TODO weight sharing?
        if params.trf_pos_emb == LEARNABLE:
            self.pos_v = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_a = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
            self.pos_t = nn.Embedding(num_embeddings=params.max_length+1, embedding_dim=params.trf_model_dim)
        elif params.trf_pos_emb == SINUSOID:
            self.pos_v = create_positional_embeddings_matrix(max_seq_len=params.max_length, dim=params.trf_model_dim)
            self.pos_a = self.pos_v
            self.pos_t = self.pos_v


        v_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.v_transformer = nn.TransformerEncoder(v_transformer_layer, num_layers=params.trf_num_v_layers)

        a_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                              batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.a_transformer = nn.TransformerEncoder(a_transformer_layer, num_layers=params.trf_num_at_layers)
        t_transformer_layer = nn.TransformerEncoderLayer(d_model=params.trf_model_dim, nhead=params.trf_num_heads,
                                                         batch_first=True, dim_feedforward=2*params.trf_model_dim)
        self.t_transformer = nn.TransformerEncoder(t_transformer_layer, num_layers=params.trf_num_at_layers)

        self.v2a_transformer = CustomTransformerEncoderLayer(input_dim = params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1, hidden_dim=2*params.trf_model_dim)
        self.v2t_transformer = CustomTransformerEncoderLayer(input_dim=params.trf_model_dim, hidden_dim=2*params.trf_model_dim,
                                                             num_heads=params.trf_num_heads, dropout=0.1)

        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(3*params.trf_model_dim, 1)

        self.pooling = nn.AvgPool2d(kernel_size=(4,1), stride=(2,1))

    def forward(self, v:torch.Tensor, a, t, mask):
        #print(v.get_device())
        #print(a.get_device())
        #print(t.get_device())
        #print(self.a_projection.weight.get_device())
        #print(self.a_projection.bias.get_device())
        v = self.tanh(self.v_projection(v))
        a = self.tanh(self.a_projection(a)) # BS, SL, dim
        t = self.tanh(self.t_projection(t)) # BS, SL, dim

        emb_indices = indices_from_mask(mask)
        v_pos = self.pos_v(emb_indices)
        a_pos = self.pos_a(emb_indices)
        t_pos = self.pos_t(emb_indices)
        v = v + v_pos
        a = a + a_pos
        t = t + t_pos


        trf_key_mask = ~mask.bool()
        trf_3d_mask = ~get_3d_mask(mask, self.num_heads).bool()
        v = self.v_transformer(v, src_key_padding_mask = trf_key_mask, mask =trf_3d_mask) # BS, SL, dim
        a = self.a_transformer(a, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL ,dim
        t = self.t_transformer(t, src_key_padding_mask = trf_key_mask, mask=trf_3d_mask) # BS, SL, dim

        a = self.v2a_transformer(query=v, key=a, value=a, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)
        t = self.v2t_transformer(query=v, key=t, value=t, key_padding_mask=trf_key_mask, attn_mask=trf_3d_mask)

        representation = torch.concatenate([v,a,t], dim=-1)
        #representation = self.pooling(representation)
        logits = self.classification(self.dropout(representation))
        logits = self.pooling(logits)
        return logits
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weight
from module import Linear
from embedding import PositionalEncoding

class RelativeMultiHeadAttention(nn.Module):
    """
    input:
    query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    """
    
    def __init__(self, d_model=512, n_heads=16, dropout_p=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = int(d_model / n_heads)
        self.n_heads = n_heads
        self.sqrt_dim = math.sqrt(d_model)
        
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_pos = Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.u_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        torch.nn.init.xavier_normal_(self.u_bias)
        torch.nn.init.xavier_normal_(self.v_bias)
        
        self.fc = Linear(d_model, d_model)
        
    def forward(self, q, k, v, pos_embedding, mask=None):
        batch_size = v.size(0)
        
        q = self.linear_q(q).view(batch_size, -1, self.n_heads, self.d_head)
        k = self.linear_k(k).view(batch_size, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.linear_v(v).view(batch_size, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        
        pos_embedding = self.linear_pos(pos_embedding).view(batch_size, -1, self.n_heads, self.d_head)
        
        content_score = torch.matmul((q + self.u_bias).transpose(1, 2), k.transpose(2, 3))
        pos_score = torch.matmul((q + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)
        
        score = (content_score + pos_score) / self.sqrt_dim
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v).transpose(1, 2)
        
        context = context.contiguous().view(batch_size, -1, self.d_model)
        
        output = self.fc(context)
        
    def _relative_shift(self, pos_score):
        batch_size, n_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, n_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        
        padded_pos_score = padded_pos_score.view(batch_size, n_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        
        return pos_score
    
# test realative shift
# def _relative_shift(pos_score):

#     batch_size, n_heads, seq_length1, seq_length2 = pos_score.size()
#     zeros = pos_score.new_zeros(batch_size, n_heads, seq_length1, 1)
#     padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
#     print(padded_pos_score)
#     padded_pos_score = padded_pos_score.view(batch_size, n_heads, seq_length2 + 1, seq_length1)
#     print(padded_pos_score)
#     pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        
#     return pos_score

# def _rel_shift(x, zero_triu=False):
#     # x: q,k,bs,n_head
#     zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
#                            device=x.device, dtype=x.dtype)
#     x_padded = torch.cat([zero_pad, x], dim=1)
#     print(x_padded)

#     x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

#     x = x_padded[1:].view_as(x)

#     return x
# ps = torch.randn(2,1,2,3)
# print(ps)
# print(_rel_shift(ps))

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, args, d_model, n_heads, dropout_p=0.1):
        super(MultiHeadedSelfAttentionModule. self).__init__()
        self.args = args
        self.positional_encoding = PositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, n_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, inputs, mask=None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length).cuda(self.args.gpu)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        inputs = self.layer_norm(inputs)
        outputs =self.attention(inputs, inputs, inputs, pos_embedding, mask)
        
        return self.dropout(outputs)
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim, dropout_p=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, query, key, value, mask=None):
        score = torch.matmul(query, key.transpose(1, 2)) / self.sqrt_dim
        
        if mask is not None:
            score.masked_fill_(mask, -np.inf)
            
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        context = torch.bmm(attn, value)
        
        return context, attn
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        
        self.attn_dim = int(d_model / n_heads)
        
        self.Linear_Q = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_K = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_V = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        init_weight(self.Linear_Q)
        init_weight(self.Linear_K)
        init_weight(self.Linear_V)
        
        self.scaled_dot_attn = ScaledDotProductAttention(self.attn_dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = v.size(0)
        query = self.Linear_Q(q).view(batch_size, -1, self.n_heads, self.attn_dim)
        key = self.Linear_K(k).view(batch_size, -1, self.n_heads, self.attn_dim)
        value = self.Linear_V(v).view(batch_size, -1, self.n_heads, self.attn_dim)
        
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
            
        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.n_heads, batch_size, -1, self.attn_dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.n_heads * self.attn_dim)
        
        return context. attn
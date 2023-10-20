import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout_rate = 0.1):
        super(ConformerBlock, self)
        self.multihead_attention = nn.MultiheadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        
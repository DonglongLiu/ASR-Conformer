import torch
import torch.nn as nn
from activation import Swish
from module import Linear

class FeedForwardModule(nn.Module):
    def __init__(self, args, encoder_dim=512, expansion_factor=4, dropout_p=0.1):
        super(FeedForwardModule, self).__init__()
        self.args = args
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.linear_a = Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.linear_b = Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.swish = Swish()
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, inputs):
        outputs = input.cuda(self.args.gpu)
        outputs = self.layer_norm(outputs)
        outputs = self.linear_a(outputs)
        outputs = self.swish(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear_b(outputs)
        outputs = self.dropout(outputs)
        
        return outputs
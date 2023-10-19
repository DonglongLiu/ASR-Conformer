import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        
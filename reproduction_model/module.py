import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            inti.zeros_(self.linear.bias)
            
    def forward(self, inputs):
        return self.linear(inputs)
    

class ResidualConnectionModule(nn.Module):
    
    """
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    
    def __init__(self, module, module_factor=1.0, input_factor=1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor
        
    def forward(self, inputs):
        residual = inputs
        module_outputs = self.module(inputs)
        
        outputs = residual * self.input_factor + module_outputs * self.module_factor
        
        return outputs
    

class LayerNorm(nn.Module):
    
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        
    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, keepdim=True)
        output = (inputs - mean) / (std + self.eps)
        output = output * self.gamma + self.beta
        
        return output
    
# ln = LayerNorm(3)
# data = torch.tensor(np.array([float(i) for i in range(12)])).resize(2,2,3)
# print(data)
# print(ln(data))
#就是做了个行的归一化，对一个样本的所有features做一个归一化，能让模型更快的收敛

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormalisation(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (self.alpha) * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.norm = LayerNormalisation()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class PositionWiseFFN(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear_2(F.relu(self.linear_1(x)))
    

class Projection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)
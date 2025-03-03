import torch
import torch.nn as nn
import torch.nn.functional as F
from TransformerComponents.AttentionHead import MultiHeadAttention
from TransformerComponents.UtilsLayers import PositionWiseFFN, ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(self, n_heads, d_model, dk, dv, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)
        self.ff =  PositionWiseFFN(d_model=d_model, d_ff=d_ff)
        self.residuals = nn.ModuleList([ResidualConnection() for _ in range(2)])
    
    def forward(self, x, mask=None):
        x = self.residuals[0](x, lambda x: self.attention(x, x, x, mask=mask))
        x = self.residuals[1](x, self.ff)
        return x
    

class Encoder(nn.Module): 
    def __init__(self, num,  n_heads, d_model, dk, dv, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(n_heads, d_model, dk, dv, d_ff) for _ in range(num)])
        # TODO: Build some LayerNormalisation Here
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x # Maybe needs to be a LayerNorm here as well
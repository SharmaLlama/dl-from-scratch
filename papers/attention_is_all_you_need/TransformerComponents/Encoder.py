import torch
import torch.nn as nn
import torch.nn.functional as F
from papers.attention_is_all_you_need.TransformerComponents.AttentionHead import MultiHeadAttention
from papers.attention_is_all_you_need.TransformerComponents.UtilsLayers import PositionWiseFFN, ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(self, n_heads, d_model, dk, dv, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)
        self.ff =  PositionWiseFFN(d_model=d_model, d_ff=d_ff)
        self.residuals = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, mask=None, return_attention=False):
        x = self.residuals[0](x, lambda x: self.attention(x, x, x, mask=mask, return_attention=return_attention))
        x = self.residuals[1](x, self.ff)
        return x
    

class Encoder(nn.Module): 
    def __init__(self, num,  n_heads, d_model, dk, dv, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(n_heads, d_model, dk, dv, d_ff, dropout) for _ in range(num)])
        # TODO: Build some LayerNormalisation Here
    def forward(self, x, mask=None, return_attention=False):
        for idx, layer in enumerate(self.layers):
            if isinstance(mask, list):
                x = layer(x, mask[idx], return_attention=return_attention)
            else: 
                x = layer(x, mask, return_attention=return_attention)
        return x # Maybe needs to be a LayerNorm here as well

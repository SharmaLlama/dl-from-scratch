import torch.nn as nn
import torch.nn.functional as F
from papers.TransformerComponents.BaseMultiHeadAttention import BaseMultiHeadAttention

class VanillaMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, n_heads, d_model, dk, dv):
        super().__init__(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)


    def attention_pattern(Q, K, V, mask=None, return_attention=False):
        dk = Q.shape[-1]
        ## Q --> batch x h x ds x dk, K --> batch x h x ds x dk, V --> batch x h x ds x dv
        attention = (Q @ K.transpose(-1, -2)) / (dk ** 0.5) # attention --> batch x h x ds x ds
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)        
        
        attention = F.softmax(attention, dim=-1)
        if return_attention:
            return attention @ V, attention, Q, K
        else:
            return attention @ V, None, None, None

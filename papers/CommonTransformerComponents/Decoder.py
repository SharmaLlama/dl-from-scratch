import torch
import torch.nn as nn
import torch.nn.functional as F
from papers.attention_is_all_you_need.VanillaAttentionHead import VanillaMultiHeadAttention
from papers.big_bird_attention.SparseAttentionHead import SparseMultiHeadAttention
from papers.CommonTransformerComponents.UtilsLayers import ResidualConnection, PositionWiseFFN
from papers.RoPE.RoPEAttentionHead import RoPEMultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, n_heads, d_model, dk, dv, d_ff, dropout, attention_type='vanilla', **kwargs):
        super().__init__()
        if attention_type == 'vanilla':
            self.attention = VanillaMultiHeadAttention(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)
            self.attention_2 = VanillaMultiHeadAttention(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)
        elif attention_type == 'sparse':
            global_tokens = kwargs.get("global_tokens", 1)
            window_tokens = kwargs.get("window_tokens", 3)
            random_tokens = kwargs.get("random_tokens", 2)
            
            self.attention = SparseMultiHeadAttention(
                n_heads=n_heads, 
                d_model=d_model, 
                dk=dk, 
                dv=dv,
                global_tokens=global_tokens,
                window_tokens=window_tokens,
                random_tokens=random_tokens
            )

            self.attention_2 = SparseMultiHeadAttention(
                n_heads=n_heads, 
                d_model=d_model, 
                dk=dk, 
                dv=dv,
                global_tokens=global_tokens,
                window_tokens=window_tokens,
                random_tokens=random_tokens
            )
        elif attention_type == 'rope':
            self.attention = RoPEMultiHeadAttention(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)
            self.attention_2 = RoPEMultiHeadAttention(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.ff =  PositionWiseFFN(d_model=d_model, d_ff=d_ff)
        self.residuals = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, encoder_mask=None, decoder_mask=None, return_attention=False):
        x = self.residuals[0](x, lambda x: self.attention(x, x, x, mask=decoder_mask, return_attention=return_attention))
        x = self.residuals[1](x, lambda x: self.attention_2(x, encoder_output, encoder_output, mask=encoder_mask, return_attention=return_attention))
        x = self.residuals[2](x, self.ff)
        return x
    
class Decoder(nn.Module):
    def __init__(self,num, n_heads, d_model, dk, dv, d_ff, dropout, attention_type='vanilla', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(n_heads, d_model, dk, dv, d_ff, dropout, attention_type, **kwargs) for _ in range(num)])
        # TODO: Build some LayerNormalisation Here
    def forward(self, x, encoder_output, encoder_mask=None, decoder_mask=None, return_attention=False):
        for idx, layer in enumerate(self.layers):
            if type(encoder_mask) == list and type(decoder_mask) == list:
                x = layer(x, encoder_output, encoder_mask[idx], decoder_mask[idx], return_attention)
            elif type(encoder_mask) == list:
                x = layer(x, encoder_output, encoder_mask[idx], decoder_mask, return_attention)
            elif type(decoder_mask) == list:
                x = layer(x, encoder_output, encoder_mask, decoder_mask[idx], return_attention)
            else:
                x = layer(x, encoder_output, encoder_mask, decoder_mask, return_attention)

        return x # Maybe needs to be a LayerNorm here as well

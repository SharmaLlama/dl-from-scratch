import torch
import torch.nn as nn
import torch.nn.functional as F

from papers.CommonTransformerComponents.Encoder import Encoder
from papers.CommonTransformerComponents.Decoder import Decoder
from papers.CommonTransformerComponents.BasePositionalEncoding import PositionalEmbedding
from papers.CommonTransformerComponents.UtilsLayers import *

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: PositionalEmbedding, tgt_embedding: PositionalEmbedding, projection: Projection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.proj = projection
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
    
    def encode(self, x, encoder_mask, return_attention=False):
        embed = self.src_embedding(x)
        return self.encoder(embed, encoder_mask, return_attention=return_attention)
    
    def decode(self, x, encoder_output, encoder_mask, decoder_mask, return_attention=False):
        embed = self.tgt_embedding(x)
        return self.decoder(embed, encoder_output, encoder_mask, decoder_mask, return_attention=return_attention)
    
    def initialise(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, encoder_mask=None, decoder_mask=None, encoder_decoder_masks=None, return_attention=False):
        x_encoder = self.encode(src, encoder_mask, return_attention=return_attention)
        if encoder_decoder_masks is not None:
            x_decoder = self.decode(tgt, x_encoder, encoder_decoder_masks, decoder_mask, return_attention=return_attention)
        else:
            x_decoder = self.decode(tgt, x_encoder, encoder_mask, decoder_mask, return_attention=return_attention)
        x = self.proj(x_decoder)
        return x

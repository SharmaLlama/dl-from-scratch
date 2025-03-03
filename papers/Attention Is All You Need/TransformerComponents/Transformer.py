import torch
import torch.nn as nn
import torch.nn.functional as F
from TransformerComponents.Encoder import Encoder
from TransformerComponents.Decoder import Decoder
from TransformerComponents.PE import PositionalEmbedding
from TransformerComponents.UtilsLayers import *

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: PositionalEmbedding, tgt_embedding: PositionalEmbedding, projection: Projection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.proj = projection
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
    
    def encode(self, x, encoder_mask):
        embed = self.src_embedding(x)
        return self.encoder(embed, encoder_mask)
    
    def decode(self, x, encoder_output, encoder_mask, decoder_mask):
        embed = self.tgt_embedding(x)
        return self.decoder(embed, encoder_output, encoder_mask, decoder_mask)
    
    def initialise(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, encoder_mask=None, decoder_mask=None):
        x_encoder = self.encode(src, encoder_mask)
        x_decoder = self.decode(tgt, x_encoder, encoder_mask, decoder_mask)
        x = self.proj(x_decoder)
        return x
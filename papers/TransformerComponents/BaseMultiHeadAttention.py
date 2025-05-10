import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseMultiHeadAttention(nn.Module, ABC):
    def __init__(self, n_heads, d_model, dk, dv):
        super().__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, n_heads * dk, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * dk, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * dv, bias=False)
        self.w_o = nn.Linear(n_heads * dv, d_model, bias=False)
        self.dk = dk
        self.dv = dv
        
    @abstractmethod
    def attention_pattern(self, Q, K, V, mask=None, return_attention=False):
        """
        Inputs are already (B, H, N, d_k / d_v).
        Must return (output, attention, Q, K) where 'attention', Q, K
        may be None unless return_attention=True
        """
        raise NotImplementedError("Subclasses must implement attention_pattern")
        
    def forward(self, Q, K, V, mask=None, return_attention=False):

        # Common preprocessing
        query = self.w_q(Q)
        key = self.w_k(K)
        value = self.w_v(V)
        batch_size, seq_length = query.shape[0], query.shape[1]
        query = query.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.n_heads, self.dv).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)
        
        # Call the attention pattern
        x, self.attention_scores, self.queries, self.keys = self.attention_pattern(
            query, key, value, mask=mask, return_attention=return_attention
        )
        
        x = x.transpose(1,2).contiguous().view(batch_size, seq_length, -1)
        x = self.w_o(x)
        return x
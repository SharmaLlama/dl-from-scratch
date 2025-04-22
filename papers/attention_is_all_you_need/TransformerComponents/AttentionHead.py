import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dk, dv):
        super().__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, n_heads * dk, bias=False) # we know that n_heads * dk or n_heads * dv is d_model 
        self.w_k = nn.Linear(d_model, n_heads * dk, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * dv, bias=False)
        self.w_o = nn.Linear(n_heads * dv, d_model, bias=False)
        self.dk = dk
        self.dv = dv
        

    @staticmethod
    def attention(Q, K, V, mask=None):
        dk = Q.shape[-1]
        ## Q --> batch x h x ds x dk, K --> batch x h x ds x dk, V --> batch x h x ds x dv
        attention = (Q @ K.transpose(-1, -2)) / (dk ** 0.5) # attention --> batch x h x ds x ds
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))        
        
        attention = F.softmax(attention, dim=-1)
        return attention @ V, attention

    def forward(self, Q, K, V, mask=None):
        query = self.w_q(Q) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        key = self.w_k(K) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        value = self.w_v(V) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        batch_size, seq_length = query.shape[0], query.shape[1]
        query = query.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.n_heads, self.dv).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask=mask)
        x = x.transpose(1,2).contiguous().view(batch_size, seq_length, -1)

        x = self.w_o(x)
        return x

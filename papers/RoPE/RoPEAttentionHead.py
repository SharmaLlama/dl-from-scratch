import torch.nn as nn
import torch.nn.functional as F
from papers.TransformerComponents.BaseMultiHeadAttention import BaseMultiHeadAttention
import torch
from papers.RoPE.RotaryEmbedding import RotaryEmbedding

class RoPEMultiHeadAttention(BaseMultiHeadAttention):
    """
    RoPEMultiHeadAttention implements a multi-head attention mechanism with Rotary Position Embedding (RoPE).
    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        dk (int): Dimensionality of the key/query vectors for each head.
        dv (int): Dimensionality of the value vectors for each head.
        max_seq_len (int, optional): Maximum sequence length for Rotary Position Embedding. Defaults to 140.
    """
    def __init__(self, n_heads, d_model, dk, dv, max_seq_len=140):
        super().__init__(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)
        self.rope = RotaryEmbedding(self.head_dim, 10_000, max_seq_len=max_seq_len)

    def attention_pattern(self, Q, K, V, mask=None, return_attention=False):
        """
        Computes the attention pattern for the given query (Q), key (K), and value (V) tensors.
        Args:
            Q (torch.Tensor): Query tensor of shape (batch, h, ds, dk), where:
                - batch: Batch size
                - h: Number of attention heads
                - ds: Sequence length
                - dk: Dimension of the key/query vectors
            K (torch.Tensor): Key tensor of shape (batch, h, ds, dk).
            V (torch.Tensor): Value tensor of shape (batch, h, ds, dv), where dv is the dimension of the value vectors.
            mask (torch.Tensor, optional): A tensor of shape (batch, h, ds, ds) used to mask certain positions in the attention scores.
                Positions with a value of 0 in the mask will be assigned a very large negative value (-1e9) in the attention scores.
                Default is None.
            return_attention (bool, optional): If True, returns the attention weights along with the output.
                Default is False.
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after applying the attention mechanism, of shape (batch, h, ds, dv).
                - torch.Tensor or None: The attention weights of shape (batch, h, ds, ds) if `return_attention` is True, otherwise None.
                - torch.Tensor or None: The query tensor (Q) if `return_attention` is True, otherwise None.
                - torch.Tensor or None: The key tensor (K) if `return_attention` is True, otherwise None.
        """
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
        
    def forward(self, Q, K, V, mask=None, return_attention=False):
        """
        Performs the forward pass of the RoPEAttentionHead with RoPE applied to queries and keys.
        Args:
            Q (torch.Tensor): The query tensor of shape (batch_size, seq_length, input_dim).
            K (torch.Tensor): The key tensor of shape (batch_size, seq_length, input_dim).
            V (torch.Tensor): The value tensor of shape (batch_size, seq_length, input_dim).
            mask (torch.Tensor, optional): An optional mask tensor of shape 
                (batch_size, n_heads, seq_length, seq_length) to apply to the attention scores. 
                Defaults to None.
            return_attention (bool, optional): If True, returns the attention scores along 
                with the output. Defaults to False.
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_length, output_dim).
            If `return_attention` is True, also returns:
                - torch.Tensor: The attention scores of shape 
                  (batch_size, n_heads, seq_length, seq_length).
        """
        query = self.w_q(Q)
        key = self.w_k(K)
        value = self.w_v(V)
        batch_size, seq_length = query.shape[0], query.shape[1]
        query = query.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.n_heads, self.dv).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)

        # rope applied Here
        query = self.rope(query)
        key = self.rope(key)

        # Call the attention pattern
        x, self.attention_scores, self.queries, self.keys = self.attention_pattern(
            query, key, value, mask=mask, return_attention=return_attention)
        
        x = x.transpose(1,2).contiguous().view(batch_size, seq_length, -1)
        x = self.w_o(x)
        return x
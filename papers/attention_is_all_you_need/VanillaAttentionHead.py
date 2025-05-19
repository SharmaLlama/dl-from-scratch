import torch.nn as nn
import torch.nn.functional as F
from papers.TransformerComponents.BaseMultiHeadAttention import BaseMultiHeadAttention

class VanillaMultiHeadAttention(BaseMultiHeadAttention):
    class VanillaMultiHeadAttention:
        """
        VanillaMultiHeadAttention is a subclass of BaseMultiHeadAttention that implements
        the multi-head attention mechanism as described in the "Attention Is All You Need" paper.
        Attributes:
            n_heads (int): The number of attention heads.
            d_model (int): The dimensionality of the input and output embeddings.
            dk (int): The dimensionality of the key and query vectors.
            dv (int): The dimensionality of the value vectors.
        """
    def __init__(self, n_heads, d_model, dk, dv):
        super().__init__(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)


    def attention_pattern(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask : torch.Tensor=None, return_attention: bool=False) -> torch.Tensor:
        """
        Computes the attention pattern for the given query (Q), key (K), and value (V) tensors.
        Args:
            Q (torch.Tensor): Query tensor of shape (batch, h, ds, dk), where:
                - batch: Batch size
                - h: Number of attention heads
                - ds: Sequence length
                - dk: Dimensionality of the query/key vectors
            K (torch.Tensor): Key tensor of shape (batch, h, ds, dk).
            V (torch.Tensor): Value tensor of shape (batch, h, ds, dv), where:
                - dv: Dimensionality of the value vectors
            mask (torch.Tensor, optional): A tensor of shape (batch, h, ds, ds) used to mask out certain positions 
                in the attention scores. Positions with a value of 0 are masked out. Default is None.
            return_attention (bool, optional): If True, returns the attention scores along with the output. 
                Default is False.
        Returns:
            torch.Tensor: The output tensor after applying the attention mechanism, of shape (batch, h, ds, dv).
            torch.Tensor or None: The attention scores of shape (batch, h, ds, ds) if `return_attention` is True, 
                otherwise None.
            torch.Tensor or None: The query tensor Q if `return_attention` is True, otherwise None.
            torch.Tensor or None: The key tensor K if `return_attention` is True, otherwise None.
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

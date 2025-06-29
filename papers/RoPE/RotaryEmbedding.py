import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    RotaryEmbedding is a PyTorch module that implements rotary positional embeddings for sequence data. 
    This technique is commonly used in transformer-based models to encode positional information into 
    input embeddings.
    Attributes:
        inv_freq (torch.Tensor): The inverse frequency tensor used to compute sinusoidal embeddings.
        max_seq_len (int): The maximum sequence length for which embeddings are precomputed.
        sin (torch.Tensor): Precomputed sine values for positional embeddings.
        cos (torch.Tensor): Precomputed cosine values for positional embeddings.
    Methods:
        __init__(dim: int, base: float = 10000.0, max_seq_len: int = 512):
            Initializes the RotaryEmbedding module.
            Args:
                dim (int): The dimensionality of the input embeddings. Must be even.
                base (float, optional): The base value for computing inverse frequencies. Default is 10000.0.
                max_seq_len (int, optional): The maximum sequence length for precomputing embeddings. Default is 512.
            Raises:
                ValueError: If `dim` is not an even number.

    """
    def __init__(self, dim, base=10000.0, max_seq_len=512):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim//2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len = max_seq_len
        sin, cos = self._build_sin_cos(max_seq_len, device=None, dtype=inv_freq.dtype)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
  
    def _build_sin_cos(self, seq_len, device= None, dtype= None):
        """
        Computes sine and cosine positional encodings for a given sequence length.

        Args:
            seq_len (int): The length of the sequence for which positional encodings are computed.
            device (torch.device | None, optional): The device on which the tensors will be allocated. 
                If None, defaults to the device of `self.inv_freq`.
            dtype (torch.dtype | None, optional): The data type of the resulting tensors. 
                If None, defaults to the dtype of `self.inv_freq`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - sin (torch.Tensor): The sine positional encodings, repeated along the last dimension.
                - cos (torch.Tensor): The cosine positional encodings, repeated along the last dimension.
        """
        device = device or self.inv_freq.device
        dtype  = dtype  or self.inv_freq.dtype

        pos = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.einsum("i,j->ij", pos, self.inv_freq).to(dtype)
        sin, cos = angles.sin(), angles.cos()
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)
        return sin, cos

    def update_max_seq_len(self, new_max_seq_len):
        """
        Update the maximum sequence length and recreate the buffers.
        """
        if new_max_seq_len != self.max_seq_len:
            self.max_seq_len = new_max_seq_len
            device = self.sin.device
            dtype = self.sin.dtype
            sin, cos = self._build_sin_cos(new_max_seq_len, device=device, dtype=dtype)
            self.register_buffer("sin", sin)
            self.register_buffer("cos", cos)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even, x_odd = x.chunk(2, dim=-1)
        return torch.cat((-x_odd, x_even), dim=-1)
    
    def forward(self, x, offset=0):
            # ensure sin/cos live on same device as x
            sin = self.sin.to(x.device) 
            cos = self.cos.to(x.device)
            if offset > 0 or x.size(-2) != self.max_seq_len:
                sin, cos = self._build_sin_cos(x.size(-2) + offset,
                                            device=x.device,
                                            dtype=x.dtype)
                sin = sin[offset:offset + x.size(-2)]
                cos = cos[offset:offset + x.size(-2)]
                
            while sin.dim() < x.dim():
                sin = sin.unsqueeze(0)
                cos = cos.unsqueeze(0)

            return (x * cos) + (self._rotate_half(x) * sin)

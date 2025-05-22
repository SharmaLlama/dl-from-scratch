import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    """
    A PyTorch module for generating positional embeddings for input sequences. 
    This module supports both learned embeddings and sinusoidal positional encodings.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied to the output embeddings.
        normal_embedding (nn.Embedding): Learnable embedding layer for input tokens.
        sin_embedding (bool): Flag to determine whether to use sinusoidal positional encoding.
        positional_encoding (torch.Tensor): Precomputed sinusoidal positional encodings.

    Args:∂
        num_embeddings (int): The size of the vocabulary or the number of unique tokens.
        embedding_dim (int): The dimensionality of the embedding vectors.
        seq_length (int): The maximum sequence length for positional encodings.
        dropout (float): Dropout probability applied to the output embeddings.
        sin_embedding (bool, optional): If True, use sinusoidal positional encoding. 
                                        If False, only use learned embeddings. Default is True.∂∂
    """
    def __init__(self, num_embeddings, embedding_dim, seq_length, dropout, sin_embedding=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normal_embedding = nn.Embedding(num_embeddings, embedding_dim) ## output is input x embedding_dim
        position = torch.arange(seq_length).unsqueeze(1)
        self.sin_embedding = sin_embedding
        positional_encoding = torch.zeros(seq_length, embedding_dim)
        positional_encoding[:, 0::2] = torch.sin(position/10_000**(torch.arange(0, embedding_dim, 2) / embedding_dim))
        positional_encoding[:, 1::2] = torch.cos(position/10_000**(torch.arange(0, embedding_dim, 2) / embedding_dim))

        self.register_buffer("positional_encoding", positional_encoding)
    def forward(self, x): # input size is batch x seq_length x d_model
        x = self.normal_embedding(x) + (self.positional_encoding if self.sin_embedding else 0) 
        return self.dropout(x)

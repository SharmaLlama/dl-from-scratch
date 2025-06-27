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
        max_seq_length (int): Maximum sequence length for precomputed encodings.
        embedding_dim (int): Dimensionality of embeddings.
    
    Args:
        num_embeddings (int): The size of the vocabulary or the number of unique tokens.
        embedding_dim (int): The dimensionality of the embedding vectors.
        max_seq_length (int): The maximum sequence length for positional encodings.
        dropout (float): Dropout probability applied to the output embeddings.
        sin_embedding (bool, optional): If True, use sinusoidal positional encoding.
                                       If False, only use learned embeddings. Default is True.
    """
    
    def __init__(self, num_embeddings, embedding_dim, max_seq_length, dropout, sin_embedding=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normal_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.sin_embedding = sin_embedding
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        
        if self.sin_embedding:
            self.register_buffer("positional_encoding", self._create_positional_encoding(max_seq_length))
    
    def _create_positional_encoding(self, seq_length):
        position = torch.arange(seq_length).unsqueeze(1).float()
        positional_encoding = torch.zeros(seq_length, self.embedding_dim)
        
        positional_encoding[:, 0::2] = torch.sin(
            position / 10_000**(torch.arange(0, self.embedding_dim, 2).float() / self.embedding_dim)
        )
        positional_encoding[:, 1::2] = torch.cos(
            position / 10_000**(torch.arange(0, self.embedding_dim, 2).float() / self.embedding_dim)
        )
        
        return positional_encoding
    
    def update_max_seq_length(self, new_max_seq_length):
        """
        Update the maximum sequence length and recreate the buffer.
        """
        if new_max_seq_length != self.max_seq_length:
            self.max_seq_length = new_max_seq_length
            if self.sin_embedding:
                device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
                new_encoding = self._create_positional_encoding(new_max_seq_length).to(device)
                self.register_buffer("positional_encoding", new_encoding)
    
    def _get_positional_encoding(self, seq_length):
        if not self.sin_embedding:
            return 0
            
        if seq_length <= self.max_seq_length:
            return self.positional_encoding[:seq_length]
        else:
            device = self.normal_embedding.weight.device
            return self._create_positional_encoding(seq_length).to(device)
    
    def forward(self, x):
        """
        Forward pass of the positional embedding layer.
        
        Args:
            x (torch.Tensor): Input token indices of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Embedded tokens with positional encoding of shape 
                         (batch_size, seq_length, embedding_dim)
        """
        assert x.dim() == 2, f"Expected 2D input (batch_size, seq_length), got {x.dim()}D"
        
        _, seq_length = x.shape
        
        token_embeddings = self.normal_embedding(x)
        pos_encoding = self._get_positional_encoding(seq_length)
        
        if isinstance(pos_encoding, torch.Tensor):
            pos_encoding = pos_encoding.to(token_embeddings.device)
        
        embeddings_with_pos = token_embeddings + pos_encoding
        
        return self.dropout(embeddings_with_pos)
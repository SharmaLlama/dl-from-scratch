import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, seq_length):
        super().__init__()
        self.normal_embedding = nn.Embedding(num_embeddings, embedding_dim) ## output is input x embedding_dim
        position = torch.arange(seq_length).unsqueeze(1)
        
        self.positional_encoding = torch.zeros(seq_length, embedding_dim)
        self.positional_encoding[:, 0::2] = torch.sin(position/10_000**(torch.arange(0, embedding_dim, 2) / embedding_dim))
        self.positional_encoding[:, 1::2] = torch.cos(position/10_000**(torch.arange(0, embedding_dim, 2) / embedding_dim))

        # self.register_buffer("positional_encoding", self.positional_encoding)
    def forward(self, x): # input size is batch x seq_length x d_model
        return self.normal_embedding(x) + self.positional_encoding
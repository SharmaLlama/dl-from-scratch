import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dk, dv, seq_len=140, global_tokens=1, window_tokens=3, random_tokens=2):
        super().__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, n_heads * dk, bias=False) # we know that n_heads * dk or n_heads * dv is d_model 
        self.w_k = nn.Linear(d_model, n_heads * dk, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * dv, bias=False)
        self.w_o = nn.Linear(n_heads * dv, d_model, bias=False)
        self.dk = dk
        self.dv = dv
        self.global_tokens = global_tokens
        self.window_tokens = window_tokens
        self.random_tokens = random_tokens

        self.seq_length = seq_len
        self.positions = torch.arange(global_tokens, seq_len - global_tokens, device="cpu")
        self.idx_tensor = self.create_idx_tensor()


    @staticmethod
    def generate_consecutive_sublists(n, l):
        """
        Generate consecutive sublists of length l containing numbers from 0 to n.
        
        Args:
            n: The maximum number (inclusive)
            l: The length of each sublist
        
        Returns:
            A PyTorch tensor with shape [num_sublists, l]
        """
        # Calculate how many valid sublists we can create
        num_sublists = n - l + 2  # +2 because n is inclusive and we start from 0
        
        indices = torch.arange(num_sublists).unsqueeze(1)
        offsets = torch.arange(l).unsqueeze(0)
        result = indices + offsets
        return result
    
    def create_idx_tensor(self):
        device = self.w_q.weight.device
        special_num = self.global_tokens  - (self.window_tokens - 1) // 2
        indices = SparseMultiHeadAttention.generate_consecutive_sublists(self.seq_length, self.window_tokens) + max(special_num, 0)
        rand_idx = []
        for i in range(self.global_tokens, self.seq_length - self.global_tokens):
            rdx_idx = []
            for _ in range(self.random_tokens):
                sampled = np.random.choice(np.arange(1, self.seq_length))
                while i - (self.window_tokens - 1) // 2 <= sampled <= i + (self.window_tokens - 1) // 2 and sampled not in rdx_idx :
                    sampled = np.random.choice(np.arange(1, self.seq_length))
                rdx_idx.append(sampled)
            rand_idx.append(rdx_idx)

        idx_list = []
        for i in range(len(self.positions)):
            row = [0] + rand_idx[i] + indices[i].tolist()
            idx_list.append(row)
        
        idx = torch.tensor(idx_list, device=device, dtype=torch.long)
        idx = idx.unsqueeze(0).unsqueeze(0) # shape: (1, 1, seq_len - g, 1 + r + w)
        return idx
    
    @staticmethod
    def attention(Q, K, V, idx_tensor,  g=1, w=3, r=2, mask=None, return_attention=False):
        """
        Computes the sparse attention mechanism for a given set of query, key, and value tensors. I chose not to just apply the masks to calculate sparseness since
        that would be cheating for the attention mechanism. The goal is to reduce the number of matrix multiplications even if the overall implementation is inefficient
        due to non-optimised kernel operators and creating of zero matrix everytime this is called. 
        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, n_heads, seq_length, dk).
            K (torch.Tensor): Key tensor of shape (batch_size, n_heads, seq_length, dk).
            V (torch.Tensor): Value tensor of shape (batch_size, n_heads, seq_length, dv).
            indices (list or torch.Tensor): List or tensor containing the indices for sparse attention.
            mask (torch.Tensor, optional): Optional mask tensor to apply on the attention scores. Defaults to None.
            return_attention (bool, optional): If True, returns the attention weights along with the output. Defaults to False.
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after applying attention, of shape (batch_size, n_heads, seq_length, dv).
                - torch.Tensor: The attention weights tensor of shape (batch_size, n_heads, seq_length, seq_length).
                - torch.Tensor or None: The query tensor (Q) if `return_attention` is True, otherwise None.
                - torch.Tensor or None: The key tensor (K) if `return_attention` is True, otherwise None.
        """
        batch_size, h = Q.shape[0], Q.shape[1]
        ds = Q.shape[-2]
        dk = Q.shape[-1]

        ## Q --> batch x h x ds x dk, K --> batch x h x ds x dk, V --> batch x h x ds x dv
        
        ## global attn
        global_idx = [i for i in range(g)] + [i for i in range(ds- g, ds)]
        global_attn = Q[:, :, global_idx, :] @ K.transpose(-1, -2)
        rand_idxs = idx_tensor[0, 0, :, 1: -w] # first one is zero and last w are the window tokens
        idx_tensor = idx_tensor.expand(batch_size, h, -1, -1)

        ## rand attn + window attn + 1st col attn
        rand_keys = K[:, :, rand_idxs, :]

        wind_idxs = idx_tensor[0, 0, :, -w:]
        
        window_keys = K[:, :, wind_idxs, :]
        first_keys = K[:, :, 0:1, :].expand(-1, -1, ds - 2 * g, -1).unsqueeze(-2)

        combined_keys = torch.cat([
            first_keys,
            rand_keys[:, :, :, :],
            window_keys
        ], dim=-2)

        attn_scores = torch.einsum("bhsd,bhskd->bhsk", Q[:, :, g:-g, :], combined_keys) ## non global attn matrix mult
        
        result = torch.zeros(
            (batch_size, h, ds, ds),
            device=Q.device,
        )
        result[:, :, global_idx, :] = global_attn
        positions = torch.arange(g, ds - g)
        temp = result[:, :, positions, :].clone()

        result[:, :, positions, :] = temp.scatter(
            dim=-1,
            index=idx_tensor,
            src=attn_scores
        )

        result /= (dk ** 0.5)
        if mask is not None:
            result = result.masked_fill(mask == 0, -1e9)

        attention= F.softmax(result, dim=-1)

        if return_attention:
            return attention @ V, result, Q, K 
        else:
            return attention @ V, result, None, None
        

    def forward(self, Q, K, V, mask=None, return_attention=False):
        query = self.w_q(Q) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        key = self.w_k(K) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        value = self.w_v(V) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        batch_size, seq_length = query.shape[0], query.shape[1]
        query = query.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.n_heads, self.dv).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)

        x, self.attention_scores, self.queries, self.keys = SparseMultiHeadAttention.attention(query, key, value, idx_tensor=self.idx_tensor, 
                                                                                               g=self.global_tokens, r=self.random_tokens, 
                                                                                               w=self.window_tokens, mask=mask, return_attention=return_attention)
        x = x.transpose(1,2).contiguous().view(batch_size, seq_length, -1)
        x = self.w_o(x)
        return x
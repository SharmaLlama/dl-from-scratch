import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import math

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
    def attention_sparse(Q, K, V, idx_tensor, g=1, w=3, r=2, mask=None,
                        return_attention=False):
        """
        Sparse attention that keeps O(B·H·N·k) memory and FLOPs.
        """
        B, H, N, d_k = Q.shape          # sequence length = N
        d_v          = V.shape[-1]
        device       = Q.device
        n_ng         = N - 2 * g        # non-global tokens
        k            = 1 + r + w        # first-token + random + window

        if g > 0:
            global_idx = torch.cat((
                torch.arange(g,      device=device),
                torch.arange(N - g,  N, device=device)
            ))                                   # (2g,)
            print(global_idx)
            Q_g = Q[:, :, global_idx, :]         # (B, H, 2g, d_k)

            logits_g  = torch.matmul(Q_g, K.transpose(-1, -2)) / math.sqrt(d_k)
            if mask is not None:
                logits_g = logits_g.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))
            prob_g = F.softmax(logits_g, dim=-1)                         # (B,H,2g,N)
            out_g  = torch.matmul(prob_g, V)                             # (B,H,2g,d_v)
        else:
            out_g = None

        if n_ng > 0:
            Q_ng = Q[:, :, g:-g, :]                                      # (B,H,n_ng,d_k)
            idx_exp = idx_tensor.expand(B, H, -1, -1)          # (B,H,n_ng,k)
            K_uns  = K.unsqueeze(3)                            # (B,H,N,1,d_k)
            idx_uns = idx_exp.unsqueeze(-1)                    # (B,H,n_ng,k,1)

            K_sel = torch.take_along_dim(K_uns, idx_uns, dim=2)     # (B,H,n_ng,k,d_k)

            V_uns  = V.unsqueeze(3)                            # (B,H,N,1,d_v)
            V_sel  = torch.take_along_dim(V_uns, idx_uns.expand(-1,-1,-1,-1,V.shape[-1]),
                                        dim=2)               # (B,H,n_ng,k,d_v)
            logits_ng = torch.einsum("bhnd,bhnkd->bhnk", Q_ng, K_sel) / math.sqrt(d_k)

            if mask is not None:
                # mask shape (B,N) → index then broadcast to logits_ng
                mask_sel = mask.gather(1, idx_exp.view(B, -1)).view(B, n_ng, k)
                logits_ng = logits_ng.masked_fill(mask_sel.unsqueeze(1) == 0, float('-inf'))

            prob_ng   = F.softmax(logits_ng, dim=-1)                      # (B,H,n_ng,k)
            out_ng    = torch.sum(prob_ng.unsqueeze(-1) * V_sel, dim=-2)  # (B,H,n_ng,d_v)
        else:
            out_ng = None

        out = torch.empty(B, H, N, d_v, device=device)
        if g > 0:
            out[:, :, 0:g,     :] = out_g[:, :, 0:g, :]
            out[:, :, N-g:N,   :] = out_g[:, :, g:,  :]
        if n_ng > 0:
            out[:, :, g:-g, :] = out_ng

        if return_attention:
            attn_dense = torch.zeros(B, H, N, N, device=device)
            if g > 0:
                attn_dense[:, :, global_idx, :] = prob_g
            if n_ng > 0:
                attn_dense[:, :, g:-g, :].scatter_(
                    dim=-1, index=idx_exp, src=prob_ng
                )
            return out, attn_dense, Q, K
        else:
            return out, None, None, None

    def forward(self, Q, K, V, mask=None, return_attention=False):
        query = self.w_q(Q) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        key = self.w_k(K) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        value = self.w_v(V) ## Batch x seq_len x d_model --> batch x seq_length x d_model
        batch_size, seq_length = query.shape[0], query.shape[1]
        query = query.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.n_heads, self.dv).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.n_heads, self.dk).transpose(1, 2)


        x, self.attention_scores, self.queries, self.keys = SparseMultiHeadAttention.attention_sparse(query, key, value, idx_tensor=self.idx_tensor, 
                                                                                               g=self.global_tokens, r=self.random_tokens, 
                                                                                               w=self.window_tokens, mask=mask, return_attention=return_attention)
        x = x.transpose(1,2).contiguous().view(batch_size, seq_length, -1)
        x = self.w_o(x)
        return x
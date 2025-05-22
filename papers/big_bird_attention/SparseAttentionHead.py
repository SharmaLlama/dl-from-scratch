from papers.CommonTransformerComponents.BaseMultiHeadAttention import BaseMultiHeadAttention
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class SparseMultiHeadAttention(BaseMultiHeadAttention):
    """
    SparseMultiHeadAttention implements a sparse attention mechanism that reduces memory and computational complexity 
    by focusing on a subset of tokens in the sequence. This class extends the BaseMultiHeadAttention class.
    Attributes:
        global_tokens (int): Number of global tokens that attend to all tokens in the sequence.
        window_tokens (int): Number of consecutive tokens attended to in a sliding window manner.
        random_tokens (int): Number of randomly selected tokens attended to for each token.
        len_cache (dict): Cache for storing precomputed index tensors for different sequence lengths and devices.
    """
    def __init__(self, n_heads, d_model, dk, dv,  global_tokens=1, window_tokens=3, random_tokens=2):
        super().__init__(n_heads, d_model, dk, dv)
        self.global_tokens = global_tokens
        self.window_tokens = window_tokens
        self.random_tokens = random_tokens
        self.len_cache = {}


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
    
    def create_idx_tensor(self, seq_length, device):
        """Create index tensor for sparse attention.
        
        Args:
            seq_length: Length of the sequence
            device: Device to create tensor on
            
        Returns:
            Index tensor of shape [1, 1, seq_len - 2*global_tokens, 1 + random_tokens + window_tokens]
        """
        special_num = self.global_tokens - (self.window_tokens - 1) // 2
        indices = SparseMultiHeadAttention.generate_consecutive_sublists(
            seq_length, self.window_tokens
        ) + max(special_num, 0)
        
        rand_idx = []
        for i in range(self.global_tokens, seq_length - self.global_tokens):
            rdx_idx = []
            for _ in range(self.random_tokens):
                sampled = np.random.choice(np.arange(1, seq_length))
                while (i - (self.window_tokens - 1) // 2 <= sampled <= i + (self.window_tokens - 1) // 2) and sampled not in rdx_idx:
                    sampled = np.random.choice(np.arange(1, seq_length))
                rdx_idx.append(sampled)
            rand_idx.append(rdx_idx)

        idx_list = []

        for i in range(len(range(self.global_tokens, seq_length - self.global_tokens))):
            if i == seq_length - 1:
                row = [0] + rand_idx[i] + [seq_length - i for i in range(self.window_tokens)] 
            else:
                row = [0] + rand_idx[i] + indices[i].tolist()

            idx_list.append(row)
        
        idx = torch.tensor(idx_list, device=device, dtype=torch.long)
        idx = idx.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len - 2*g, 1 + r + w)
        return idx
    
           
    def attention_pattern(self, Q, K, V, mask=None, return_attention=False):
        """
        Implements sparse attention with reduced memory and computational complexity.
        This method computes attention using a sparse pattern that includes global tokens,
        random tokens, and window tokens. It reduces the memory and computational requirements
        to O(B·H·N·k), where B is the batch size, H is the number of attention heads, N is the
        sequence length, and k is the number of tokens attended to by each token.
        Args:
            Q (torch.Tensor): Query tensor of shape (B, H, N, d_k), where B is the batch size,
                H is the number of attention heads, N is the sequence length, and d_k is the
                dimensionality of the query/key vectors.
            K (torch.Tensor): Key tensor of shape (B, H, N, d_k).
            V (torch.Tensor): Value tensor of shape (B, H, N, d_v), where d_v is the
                dimensionality of the value vectors.
            mask (torch.Tensor, optional): Attention mask tensor. Can have one of the following
                shapes:
                - (B, 1, 1, N): Broadcastable mask for all heads and tokens.
                - (B, 1, N, N): Mask for all heads with specific token-to-token masking.
                Defaults to None.
            return_attention (bool, optional): If True, returns the attention weights along
                with the output. Defaults to False.
        Returns:
            tuple:
                - out (torch.Tensor): Output tensor of shape (B, H, N, d_v) containing the
                  attended values.
                - attn_dense (torch.Tensor or None): Dense attention weights of shape
                  (B, H, N, N) if `return_attention` is True, otherwise None.
                - Q (torch.Tensor or None): The input query tensor, returned as-is if
                  `return_attention` is True, otherwise None.
                - K (torch.Tensor or None): The input key tensor, returned as-is if
                  `return_attention` is True, otherwise None.
        Notes:
            - The method handles both global and non-global tokens separately.
            - Global tokens attend to all tokens, while non-global tokens attend to a
              subset of tokens determined by the sparse attention pattern.
            - The method supports causal and non-causal masks.
        """

        B, H, N, d_k = Q.shape          # sequence length = N
        d_v = V.shape[-1]
        device = Q.device
        g = self.global_tokens
        w = self.window_tokens
        r = self.random_tokens
        n_ng = N - 2 * g                # non-global tokens
        k = 1 + r + w                   # first-token + random + window
        
        if N not in self.len_cache:
            self.len_cache[N] = {}               
        if device not in self.len_cache[N]:
            with torch.no_grad():
                self.len_cache[N][device] = self.create_idx_tensor(N, device)

        idx_tensor = self.len_cache[N][device]     
        
        # Process global tokens
        if g > 0:
            global_idx = torch.cat((
                torch.arange(g, device=device),
                torch.arange(N - g, N, device=device)
            ))                                  # (2g,)
            Q_g = Q[:, :, global_idx, :]        # (B, H, 2g, d_k)
            
            logits_g = torch.matmul(Q_g, K.transpose(-1, -2)) / math.sqrt(d_k)
            
            # Handle mask for global tokens
            if mask is not None:
                # Check mask shape and adapt accordingly
                if mask.dim() == 4:
                    if mask.shape[1] == 1 and mask.shape[2] == 1:  # B x 1 x 1 x seq_len
                        logits_g = logits_g.masked_fill(mask == 0, -1e9)
                    elif mask.shape[1] == 1:  # B x 1 x seq_len x seq_len
                        mask_g = mask[:, :, global_idx, :]
                        logits_g = logits_g.masked_fill(mask_g == 0, -1e9)
            
            prob_g = F.softmax(logits_g, dim=-1)  # (B,H,2g,N)
            out_g = torch.matmul(prob_g, V)       # (B,H,2g,d_v)
        else:
            out_g = None
            prob_g = None
        
        # Process non-global tokens
        if n_ng > 0:
            Q_ng = Q if g == 0 else Q[:, :, g:-g, :]                # (B,H,n_ng,d_k)
            idx_exp = idx_tensor.expand(B, H, -1, -1)  # (B,H,n_ng,k)
            K_uns = K.unsqueeze(3)   # (B,H,N,1,d_k)
            idx_exp = torch.clamp(idx_exp, 0, N-1)
            idx_uns = idx_exp.unsqueeze(-1)        # (B,H,n_ng,k,1)
            K_sel = torch.take_along_dim(K_uns, idx_uns, dim=2)  # (B,H,n_ng,k,d_k)
            
            V_uns = V.unsqueeze(3)                 # (B,H,N,1,d_v)
            V_sel = torch.take_along_dim(
                V_uns, 
                idx_uns.expand(-1, -1, -1, -1, V.shape[-1]),
                dim=2
            )  # (B,H,n_ng,k,d_v)
            
            logits_ng = torch.einsum("bhnd,bhnkd->bhnk", Q_ng, K_sel) / math.sqrt(d_k)
            
            # Handle mask for non-global tokens
            if mask is not None:
                if mask.dim() == 4:
                    if mask.shape[1] == 1 and mask.shape[2] == 1:  # B x 1 x 1 x seq_len
                        # For causal masks: gather the relevant values from the mask
                        mask_indices = idx_exp.reshape(B, H * n_ng * k)
                        gathered_mask = torch.gather(
                            mask.squeeze(1).squeeze(1), 
                            1, 
                            mask_indices
                        ).reshape(B, H, n_ng, k)
                        logits_ng = logits_ng.masked_fill(gathered_mask == 0, -1e9)
                    elif mask.shape[1] == 1:  # B x 1 x seq_len x seq_len
                        mask_bh   = mask.squeeze(1).unsqueeze(1)   # (B,1,N,N) → broadcast later
                        if mask_bh.size(1) != H:
                            mask_bh = mask_bh.expand(-1, H, -1, -1)

                        mask_rows = mask_bh[:, :, g:-g, :]         # (B,H,n_ng,N)
                        gathered  = torch.gather(mask_rows, -1, idx_exp)  # (B,H,n_ng,k)
                        logits_ng = logits_ng.masked_fill(gathered == 0, -1e9)
                                
            prob_ng = F.softmax(logits_ng, dim=-1)  # (B,H,n_ng,k)
            out_ng = torch.sum(prob_ng.unsqueeze(-1) * V_sel, dim=-2)  # (B,H,n_ng,d_v)
        else:
            out_ng = None
            prob_ng = None
        
        out = torch.zeros(B, H, N, d_v, device=device)
        if g > 0:
            out[:, :, 0:g, :] = out_g[:, :, 0:g, :]
            out[:, :, N-g:N, :] = out_g[:, :, g:, :]
        if n_ng > 0:
            if g > 0:
                out[:, :, g:-g, :] = out_ng
            else:
                out[:, :, :, :] = out_ng

        
        if return_attention:
            attn_dense = torch.zeros(B, H, N, N, device=device)
            if g > 0:
                attn_dense[:, :, global_idx, :] = prob_g
            if n_ng > 0:
                if g > 0:
                    attn_dense[:, :, g:-g, :].scatter_(
                    dim=-1, index=idx_exp, src=prob_ng)
                else:
                    attn_dense[:, :, :, :].scatter_(
                        dim=-1, index=idx_exp, src=prob_ng)
            return out, attn_dense, Q, K
        else:
            return out, None, None, None
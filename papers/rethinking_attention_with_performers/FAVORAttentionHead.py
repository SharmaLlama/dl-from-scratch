import torch
import torch.nn.functional as F
import numpy as np
np.random.seed(42)
import os
from papers.CommonTransformerComponents.BaseMultiHeadAttention import BaseMultiHeadAttention

class FAVORMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, n_heads, d_model, dk, dv, max_seq_len=140, m_features=256, kernel_type="trig", 
                 use_orthogonal=True):
        super().__init__(n_heads=n_heads, d_model=d_model, dk=dk, dv=dv)
        self.m_features = m_features
        self.kernel_type = kernel_type
        self.use_orthogonal = use_orthogonal
        self.register_buffer('omegas', self._create_random_projections())

    def _create_random_projections(self):
        """Create random projection matrices for each head"""
        if self.use_orthogonal:
            # Stack orthogonal matrices for each head
            omegas_list = []
            for _ in range(self.n_heads):
                omega = self._orthogonal_gaussian(self.m_features, self.dk)
                omegas_list.append(omega)
            return torch.stack(omegas_list, dim=0)  # (num_heads, m_features, d_head)
        else:
            # IID Gaussian for each head
            return torch.randn(self.n_heads, self.m_features, self.dk)
        
    def _orthogonal_gaussian(self, m, d):
        """Generate orthogonal random matrix for one head"""
        def orthogonal_square():
            q, _ = torch.linalg.qr(torch.randn(d, d))
            return q.T
        
        num_squares = m // d
        blocks = [orthogonal_square() for _ in range(num_squares)]
        
        remainder = m - d * num_squares
        if remainder > 0:
            blocks.append(orthogonal_square()[:remainder])
        
        matrix = torch.cat(blocks, dim=0)
        divisor = np.sqrt(num_squares + remainder / d)
        return matrix / divisor

    def _random_feature_map(self, x, omegas):
        """
        Compute random features for multi-head case
        Args:
            x: (B, num_heads, L, d_head)
            omegas: (num_heads, m_features, d_head)
        Returns:
            features: (B, num_heads, L, feature_dim)
        """
        B, num_heads, L, d_head = x.shape
        # Ensure omegas is on the same device as x
        omegas = omegas.to(x.device)

        if self.kernel_type == "+":
            norm_sq = torch.sum(x**2, dim=-1, keepdim=True)  # (B, num_heads, L, 1)
            # Scaled projection: (num_heads, d_head, m_features)
            # Unsqueeze omegas to allow broadcasting across batch dimension
            scaled_omegas = omegas.transpose(-2, -1).unsqueeze(0) / np.sqrt(2.0)
            # (B, num_heads, L, d_head) @ (1, num_heads, d_head, m_features) -> (B, num_heads, L, m_features)
            ws = torch.matmul(x, scaled_omegas)
            f1 = torch.exp(ws)
            num_term = torch.exp(-norm_sq / 2) / np.sqrt(self.m_features)
            return num_term * f1
            
        elif self.kernel_type == "trig":
            norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
            scaled_omegas = omegas.transpose(-2, -1).unsqueeze(0) / np.sqrt(2.0)
            ws = torch.matmul(x, scaled_omegas)
            f1, f2 = torch.sin(ws), torch.cos(ws)
            num_term = torch.exp(norm_sq / 2) / np.sqrt(self.m_features)
            return num_term * torch.cat([f1, f2], dim=-1)  # (B, num_heads, L, 2*m_features)
            
        elif self.kernel_type == "hyp":
            norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
            scaled_omegas = omegas.transpose(-2, -1).unsqueeze(0) / np.sqrt(2.0)
            ws = torch.matmul(x, scaled_omegas)
            f1, f2 = torch.exp(ws), torch.exp(-ws)
            num_term = torch.exp(-norm_sq / 2) / np.sqrt(2.0 * self.m_features)
            return num_term * torch.cat([f1, f2], dim=-1)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        

    def attention_pattern(self, Q, K, V, mask=None, return_attention=False):
        """
        Performs multi-head FAVOR attention.
        
        Args:
            Q (torch.Tensor): Query tensor (B, H, L, d_head).
            K (torch.Tensor): Key tensor (B, H, L, d_head).
            V (torch.Tensor): Value tensor (B, H, L, d_head).
            mask (torch.Tensor, optional): Attention mask.
        
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after applying the attention mechanism, of shape (batch, h, ds, dv).
                - torch.Tensor or None: The attention weights of shape (batch, h, ds, ds) if `return_attention` is True, otherwise None.
                - torch.Tensor or None: The query tensor (Q) if `return_attention` is True, otherwise None.
                - torch.Tensor or None: The key tensor (K) if `return_attention` is True, otherwise None.
        """
        B, H, L, d_head = Q.shape
        query_scaled_input = Q / (d_head ** 0.25)
        key_scaled_input = K / (d_head ** 0.25)

        # Compute random features for Q and K
        new_query = self._random_feature_map(query_scaled_input, self.omegas) # (B, H, L, feature_dim)
        new_key = self._random_feature_map(key_scaled_input, self.omegas)     # (B, H, L, feature_dim)

        ones_tensor = torch.ones(B, H, L, 1, device=Q.device)
        C = torch.cat([V, ones_tensor], dim=-1)  # (B, H, L, d_head + 1)
        
        if mask is None:
            # new_key: (B, H, L, feature_dim) -> transpose to (B, H, feature_dim, L) for the first matmul
            # C: (B, H, L, d_head + 1)
            # Result: (B, H, feature_dim, d_head + 1)
            buf_1 = torch.einsum('bhlf,bhld->bhfd', new_key, C) # (B, H, feature_dim, d_head + 1)
            
            # new_query: (B, H, L, feature_dim)
            # buf_1: (B, H, feature_dim, d_head + 1)
            # Result: (B, H, L, d_head + 1)
            buf_2 = torch.einsum('bhlf,bhfd->bhld', new_query, buf_1) # (B, H, L, d_head + 1)

            # Split the result
            buf_3 = buf_2[..., :-1]  # (B, H, L, d_head) - values
            buf_4 = buf_2[..., -1]   # (B, H, L) - normalisation terms
            ans = buf_3 / (buf_4.unsqueeze(-1) + 1e-6)  # (B, H, L, d_head)
            if return_attention:
                attention_vals = torch.einsum("bhlf,bhLf->bhlL", new_query, new_key)
                return ans, attention_vals / (buf_4.unsqueeze(-1) + 1e-6), new_query, new_key
            else:
                return ans, None, None, None
        else: # this is for upper-triangular mask only
            # Unidirectional attention with prefix sum
            # G_temp: (B, H, L, feature_dim, d_head + 1)
            G_temp = torch.einsum('bhlf,bhld->bhlfd', new_key, C)

            # G: (B, H, L, feature_dim, d_head + 1) - cumulative sum along the sequence length (L)
            G = torch.cumsum(G_temp, dim=2) # Perform cumulative sum along the sequence dimension (dim=2)

            # new_query: (B, H, L, feature_dim)
            # G: (B, H, L, feature_dim, d_head + 1)
            # Result: (B, H, L, d_head + 1)
            buf_2 = torch.einsum('bhlf,bhlfd->bhld', new_query, G)
            buf_3 = buf_2[..., :-1]  # (B, H, L, d_head) - values
            buf_4 = buf_2[..., -1]   # (B, H, L) - normalisation terms

            ans = buf_3 / (buf_4.unsqueeze(-1) + 1e-6) # (B, H, L, d_head)
            
            if return_attention:
                attention_vals = torch.einsum("bhlf,bhLf->bhlL", new_query, new_key)
                return ans, G, new_query, new_key
            else:
                return ans, None, None, None

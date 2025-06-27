
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, FFN=1024, experts=8, capacity_factor=1.0, top_k=1):
        self.capacity_factor = capacity_factor
        self.experts = experts
        self.FFN = FFN
        self.top_k = top_k

    def forward(self, x):
        """
        Forward pass through the MoE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            torch.Tensor: Output tensor after passing through the MoE layer.
        """
        batch_size, seq_length, d_model = x.shape
        # Compute gating scores for each expert
        gate_scores = torch.randn(batch_size, self.experts)
        # Select top-k experts for each input
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        # Create a mask for the selected experts
        mask = torch.zeros(batch_size, self.experts, device=x.device)
        mask.scatter_(1, top_k_indices, 1.0)
        # Compute the capacity for each expert
        capacity = int(self.capacity_factor * batch_size / self.experts)
        # Create a mask to ensure each expert does not exceed its capacity
        expert_capacity_mask = mask.sum(dim=0) < capacity
        # Apply the expert capacity mask
        mask = mask * expert_capacity_mask.float()
        # Compute the output for each expert
        expert_outputs = torch.randn(batch_size, self.experts, seq_length, d_model)
        # Apply the mask to the expert outputs
        masked_outputs = expert_outputs * mask.unsqueeze(-1)
        # Sum the outputs of the selected experts
        output = masked_outputs.sum(dim=1)
        return output
    
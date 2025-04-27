import time
import torch
import torch.profiler as profiler
import numpy as np
import torch.nn.functional as F

def attention(query, key, value, rand_idx, indices, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    rand_idx_tensor = torch.tensor(rand_idx, device=device, dtype=torch.long)
    batch_size, n_heads = query.shape[0], query.shape[1]
    seq_length = query.shape[2]
    positions = torch.arange(1, seq_length - 1, device=device)
    
    indices_tensor = torch.tensor([idx.tolist() if isinstance(idx, torch.Tensor) else idx 
                                 for idx in indices], device=device, dtype=torch.long)
    
    first_row = query[:, :, [0, seq_length - 1], :] @ key.transpose(-1, -2)
    rand_vals = key[:, :, rand_idx_tensor, :]
    window_keys = torch.cat([
        key[:, :, 0:seq_length-2, :],
        key[:, :, 1:seq_length-1, :],
        key[:, :, 2:seq_length, :]
    ], dim=-1).reshape(batch_size, n_heads, seq_length-2, 3, -1)
    first_keys = key[:, :, 0:1, :].expand(-1, -1, seq_length - 2, -1)
    combined_keys = torch.cat([
        first_keys.unsqueeze(-2),
        rand_vals[:, :, :seq_length-2, :].unsqueeze(-2),
        window_keys
    ], dim=-2)
    attn_scores = torch.einsum("bhsd,bhskd->bhsk", query[:, :, 1:-1, :], combined_keys)
    
    result = torch.zeros(
        (batch_size, n_heads, seq_length, seq_length),
        device=device,
    )
    result[:, :, [0, seq_length - 1], :] = first_row
    idx_list = []
    for i in range(len(positions)):
        row = [0, rand_idx_tensor[i].item()] + indices_tensor[i].tolist()
        idx_list.append(row)
    
    idx = torch.tensor(idx_list, device=device, dtype=torch.long)
    idx = idx.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1, -1)
    
    # Use scatter for efficient assignment while maintaining gradients
    temp = result[:, :, positions, :].clone()
    result[:, :, positions, :] = temp.scatter(
        dim=-1,
        index=idx,
        src=attn_scores
    )
    
    softie = F.softmax(result, dim=-1)
    return softie @ value



def get_function_runtime(func, *args, **kwargs):
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    ) as prof:
        with profiler.record_function("function_timing"):
            start = time.time_ns()
            func(*args, **kwargs)
            end = time.time_ns()
    
    # Extract just the timing information
    events = prof.key_averages()
    for evt in events:
        if evt.key == "function_timing":
            cpu_time = evt.cpu_time_total  # Convert to milliseconds
            cuda_time = evt.cuda_time_total if hasattr(evt, "cuda_time_total") else 0
    
    return cpu_time, cuda_time, end - start


def log_space_sequence(start, end, num_points):
    log_values = np.logspace(np.log10(start), np.log10(end), num_points)
    return np.unique(np.round(log_values).astype(int))


def reg_attention(Q, K, V, mask=None, return_attention=False):
    # Determine device and ensure tensors are on it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure tensors are on the same device (no-op if already on the right device)
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    
    # Get dimension of keys
    dk = Q.shape[-1]
    
    # Compute attention scores
    # Q --> batch x h x ds x dk, K --> batch x h x ds x dk, V --> batch x h x ds x dv
    attention = (Q @ K.transpose(-1, -2)) / (dk ** 0.5)  # attention --> batch x h x ds x ds
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.to(device)  # Ensure mask is on the same device
        attention = attention.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(attention, dim=-1)
    
    # Apply attention weights to values
    output = attention_weights @ V
    
    # Return attention weights if requested
    if return_attention:
        return output, attention_weights
    
    return output

if __name__ == "__main__":
    d_model = 512
    n_heads = 4
    dk = dv =  d_model // n_heads
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    cpu_timings_sparse = []
    cpu_timings_reg = []

    batch_sizes = 2**np.arange(1, 11)
    seq_lengths = log_space_sequence(10, 512, 30)
    cpu_timings_sparse = {batch_size : [] for batch_size in batch_sizes}
    cpu_timings_reg = {batch_size : [] for batch_size in batch_sizes}
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            q = torch.rand(size=(batch_size, n_heads, seq_length, dk))
            k = torch.rand(size=(batch_size, n_heads, seq_length, dk))
            v = torch.rand(size=(batch_size, n_heads, seq_length, dv))

            rand_idx = []
            for i in range(1, seq_length - 1):
                sampled = np.random.choice(np.arange(1, seq_length))
                while i - 1 <= sampled <= i + 1:
                    sampled = np.random.choice(np.arange(1, seq_length))
                rand_idx.append(sampled)

            window_size = 3
            base = torch.arange(0, seq_length)
            indices = base.unfold(0, window_size, 1)
            cpu_timings_sparse[batch_size].append(get_function_runtime(attention, q, k,v, rand_idx, indices))
            cpu_timings_reg[batch_size].append(get_function_runtime(reg_attention, q, k, v))

    print(cpu_timings_reg)
    print("-----------------------------------------------------------")
    print(cpu_timings_sparse)
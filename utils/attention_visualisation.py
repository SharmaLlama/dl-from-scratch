from TransformerComponents.AttentionHead import MultiHeadAttention
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F

def get_attention(model, sample_batch, device):
    attention_scores_dict = {}

    def save_attention_scores(name):
        def hook(module, input, output):
            attention_scores_dict[name] = module.attention_scores
        return hook

    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            module.register_forward_hook(save_attention_scores(name))

    encoder_input = sample_batch['src'].to(device)
    tgt_input = sample_batch['tgt'].to(device)
    encoder_mask = sample_batch['encoder_mask'].to(device)
    decoder_mask = sample_batch['decoder_mask'].to(device)

    model(encoder_input, tgt_input, encoder_mask=encoder_mask, decoder_mask=decoder_mask)
    return attention_scores_dict

def process_and_visualise_attention(attentions, pixel_values, patch_size, threshold=0.6, raw_attn=False):    
    # Extract the CLS token attention (first token attending to all patches)    
    nh = attentions.shape[0]  # Number of attention heads
    attentions = attentions[:, 0, 1:].reshape(nh, -1)
    
    w_featmap = pixel_values.shape[-2] // patch_size
    h_featmap = pixel_values.shape[-1] // patch_size
    
    val, idx = torch.sort(attentions)  # Sort values per query
    val /= torch.sum(val, dim=1, keepdim=True)  # Normalize each col to sum to 1
    cumval = torch.cumsum(val, dim=1)  # Compute cumulative sum
    th_attn = cumval > (1 - threshold)  # Mask for top 'threshold' percentage
    idx2 = torch.argsort(idx)
    
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]

    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    th_attn = nn.functional.interpolate(
        th_attn.unsqueeze(0), scale_factor=patch_size, mode="bicubic"
    )[0].cpu().detach().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=patch_size, mode="bicubic"
    )[0].cpu().detach().numpy()

    image_np = pixel_values.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0,1]
    
    num_rows = 6 if raw_attn else 3
    num_cols = 4
    figsize = (num_cols * 4, num_rows * 3) 

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    
    for head in range(12):  
        row = head // num_cols 
        col = head % num_cols  
        
        ax = axes[row, col]
        ax.imshow(image_np)
        ax.imshow(th_attn[head], cmap="jet", alpha=0.4)
        ax.set_title(f"Head {head}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()



def attention_rollout(attention, pixel_values, patch_size=16, discard_ratio = 0.8, head_fusion="mean", plot=False): # attention --> layer x (batch x num_heads x seq_len x seq_len)
    result = torch.eye(attention[0].size(-1), device=attention[0].device)
    with torch.no_grad():
        for att in attention:
            if head_fusion == "mean":
                attention_fused = att.mean(axis=1) # (batch x seq_len x seq_len)
            elif head_fusion == "max":
                attention_fused = att.max(dim=1)[0] # (batch x seq_len x seq_len)
            elif head_fusion == "min":
                attention_fused = att.min(dim=1)[0] # (batch x seq_len x seq_len)
            else:
                raise "fusion method not supported"
        
            flat = attention_fused.view(att.size(0), -1) # (batch x seq_len^2)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
            indices = indices[indices != 0] # making sure we don't discard the CLS token information
            flat[0, indices] = 0
            I = torch.eye(attention_fused.size(-1), device=attention_fused.device)
            a = (attention_fused + 1.0 * I) / 2 ## due to the view, we indirectly modify the attention_fused 
            a /= a.sum(dim=-1, keepdim=True)
            result = a @ result
    
        
    mask = result[:, 0, 1:]
    mask = mask.reshape(-1, pixel_values.shape[0] // patch_size, pixel_values.shape[1] // patch_size)
    mask /= torch.max(mask)
    
    mask = nn.functional.interpolate(
        mask.unsqueeze(0), scale_factor=patch_size, mode="bilinear"
    )[0]

    if plot:
        image_np = pixel_values.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0,1]
        _, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        axes.imshow(image_np)
        axes.imshow(mask.squeeze(0).cpu().detach().numpy(), cmap="jet", alpha=0.4)
        plt.show()
    else: 
        return mask.squeeze(0)
    

def visualise_all_heads(attention, image, patch_size=16):
    num_heads = attention.shape[0]
    n_cols = int(np.ceil(np.sqrt(num_heads)))
    n_rows = int(np.ceil(num_heads / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    
    if n_rows == 1 or n_cols == 1:
        axes = np.atleast_2d(axes)
    
    grid_h = image.shape[0] // patch_size
    grid_w = image.shape[1] // patch_size
    num_patches = grid_h * grid_w
    
    for head in range(num_heads):
        att = attention[head, :, :] 
        att = att[1:, 1:]
        # Average over the query tokens (rows) to get one scalar per patch (key).
        att_map = att.mean(dim=0).detach().cpu().numpy()  # shape: [num_patches]
        
        if att_map.shape[0] != num_patches:
            print("Warning: Expected {} patches, but got {}.".format(num_patches, att_map.shape[0]))
        
        att_map = att_map.reshape((grid_h, grid_w))
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        att_map_up = cv2.resize(att_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        row = head // n_cols
        col = head % n_cols
        ax = axes[row, col]
        ax.imshow(image)
        ax.imshow(att_map_up, cmap='jet', alpha=0.4)
        ax.set_title(f"Head {head}")
        ax.axis('off')
    
    for head in range(num_heads, n_rows * n_cols):
        row = head // n_cols
        col = head % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def animate_rollout(attention, image_tensor, method="mean"):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np) 
    initial_mask =  attention_rollout(attention, image_np, discard_ratio=0.0, head_fusion=method).cpu().numpy()
    overlay_im = ax.imshow(initial_mask, cmap="jet", alpha=0.4, animated=True)
    # Range of discard_ratio values
    discard_ratios = torch.linspace(0.0, 0.95, steps=30).tolist() + [0.95] * 8

    def update(frame):
        discard_ratio = discard_ratios[frame % len(discard_ratios)]  
        mask = attention_rollout(attention, image_np, discard_ratio=discard_ratio, head_fusion=method)
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        overlay_im.set_data(mask_np)
        ax.set_title(f"Discard Ratio: {discard_ratio:.2f}, method: {method}")
        return [overlay_im]


    anim = animation.FuncAnimation(fig, update, frames=len(discard_ratios), interval=200, repeat=True)
    return anim
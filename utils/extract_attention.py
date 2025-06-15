import torch

class AttentionScoreHook:
    def __init__(self, model):
        self.model = model
        self.attention_scores = []
        self.queries = []
        self.keys = []
        self.hook_handles = []
        self.layer_count = 0
        self._register_hooks()
        
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if hasattr(module, 'attention_scores'):
                if module.queries is not None and module.keys is not None:
                    self.attention_scores.append((layer_idx, module.attention_scores.detach().cpu().clone()))
                    self.queries.append((layer_idx, module.queries.detach().cpu().clone()))
                    self.keys.append((layer_idx, module.keys.detach().cpu().clone()))
        return hook
    
    def _register_hooks(self):
        """Register hooks on all MultiHeadAttention modules"""
        layer_idx = 0
        for encoder_layer in self.model.encoder.layers:
            handle = encoder_layer.attention.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hook_handles.append(handle)
            layer_idx += 1
            self.layer_count += 1
            
        for decoder_layer in self.model.decoder.layers:
            # Self-attention
            handle = decoder_layer.attention.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hook_handles.append(handle)
            layer_idx += 1
            self.layer_count += 1
            
            # Cross-attention
            handle = decoder_layer.attention_2.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hook_handles.append(handle)
            layer_idx += 1
            self.layer_count += 1
    
    def get_attention_scores(self):
        """
        Get attention scores as [batch_size, num_layers, num_heads, seq_len, seq_len]
        
        Returns:
            Tuple of (attention_scores, queries, keys) each with shape 
            [batch_size, num_layers, num_heads, seq_len, seq_len] for attention_scores
            and appropriate shapes for queries and keys
        """
        sorted_scores = sorted(self.attention_scores, key=lambda x: x[0])
        sorted_queries = sorted(self.queries, key=lambda x: x[0])
        sorted_keys = sorted(self.keys, key=lambda x: x[0])
        
        results = []
        for sorted_tensor in [sorted_scores, sorted_queries, sorted_keys]:
            tensors = [tensor for _, tensor in sorted_tensor]
            
            # Stack along a new dimension (layer dimension)
            stacked = torch.stack(tensors, dim=0)  # [num_layers, batch, heads, seq_len, seq_len/dk/dk]
            
            # Transpose to get [batch_size, num_layers, num_heads, seq_len, seq_len/dk/dk]
            transposed = stacked.permute(1, 0, 2, 3, 4)
            result = transposed.numpy()
            results.append(result)
        
        self.attention_scores = []
        self.keys = []
        self.queries = []
        
        return results[0], results[1], results[2]
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

def extract_attention_weights(model, src, tgt, src_masks=None, tgt_masks=None):
    """
    Extract attention weights from the model.
    
    Returns:
        Tuple of (attention_weights, queries, keys) each with shape 
        [batch_size, num_layers, num_heads, seq_len, seq_len] for attention_weights
    """
    hook = AttentionScoreHook(model)    
    with torch.inference_mode():
        model(src, tgt, src_masks, tgt_masks, return_attention=True)
    
    attention_weights, queries, keys = hook.get_attention_scores()
    hook.remove_hooks()
    
    return attention_weights, queries, keys

def get_token_until_eos(sentences):
    trimmed = []
    for tokens in sentences:
        if '<EOS>' in tokens:
            eos_idx = tokens.index('<EOS>')
            trimmed.append(tokens[: eos_idx + 1])
        else:
            trimmed.append(tokens)
    return trimmed
"""
Positional Encoding Components for Transformers

NOTE:
RoPE encodes position by rotating Q and K vectors during attention.

Rotation angles depend on position, so attention naturally captures relative distances between tokens.

The Embeddings class here only handles token embeddings.

nn.embedding is an lookup table mapping token IDs to dense vectors.

So each token is mapped to a unique vector in d_model-dimensional space.

(Note:Proper weight update for nn.Embedding happens during backpropagation in training.)
"""

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: Rotary Positional Embedding"""
    
    def __init__(self, dim, base=10000):
        super().__init__() #Initialize the parent nn.Module class
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq) #This is will saved as a buffer,(essentially bufffer are saved tensors but not trained parameters)

    def forward(self, x, seq_len):                #forward is caled during the forward pass of the model 
        positions = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)
        x_rotated = x_complex * freqs_complex
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        
        return x_out.type_as(x)


class Embeddings(nn.Module):
    """Token embeddings only - RoPE handles positions"""
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        return self.token_embed(token_ids)
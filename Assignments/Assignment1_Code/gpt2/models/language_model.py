"""
Language Model - Complete Transformer Architecture
"""

import torch.nn as nn
from .embeddings import Embeddings
from .transformer_block import TransformerBlock


class LanguageModel(nn.Module):
    """Decoder-only transformer language model"""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()

        self.embeddings = Embeddings(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Weight tying - share embeddings with output
        self.lm_head.weight = self.embeddings.token_embed.weight

    def forward(self, token_ids):
        x = self.embeddings(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits
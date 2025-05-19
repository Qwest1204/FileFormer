import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, token_embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.embedding = nn.Embedding(vocab_size, token_embed_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.token_embed_dim)
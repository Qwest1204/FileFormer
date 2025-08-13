import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, vocab, emb_dim, hash_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb_dim)
        self.gru = nn.GRU(emb_dim, hash_dim, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hash_dim)

    def forward(self, chunk):  # chunk: [batch, 256000]
        embedded = self.embedding(chunk)  # [batch, 256000, embed_dim]
        _, hidden = self.gru(embedded)  # hidden: [1, batch, hash_dim]
        hash_emb = self.norm(hidden[-1])  # [batch, hash_dim]
        return hash_emb
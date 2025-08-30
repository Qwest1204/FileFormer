import torch
import torch.nn as nn
from notus.transformer_model.arch.attention import SelfAttention, MultiHeadAttention, MultiQueryAttention
from notus.transformer_model.arch.mlp import MLP
import torch.nn.functional as F
import math

class DecoderLayer(nn.Module):
    def __init__(self,
                 dim_ff: int,
                 num_heads: int,
                 embedding_dim: int,
                 activation_type: str,
                 dropout: float,
                 ):
        super(DecoderLayer, self).__init__()
        # define attention
        self.head_dim = embedding_dim // num_heads
        self.self_attention = MultiQueryAttention(embedding_dim, num_heads)
        self.cross_attention = MultiQueryAttention(embedding_dim, num_heads)
        #define mpl
        self.mlp = MLP(embedding_dim, dim_ff, activation_type, dropout)
        #define normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, padding_mask=None):
        self_attention_out = self.self_attention(x, x, x, mask=padding_mask)
        self_attention_out = self.dropout(self_attention_out)

        x = self.norm1(x + self_attention_out)

        cross_attention_out = self.cross_attention(x, context, context, mask=padding_mask)
        cross_attention_out = self.dropout(cross_attention_out)

        x = self.norm2(x + cross_attention_out)

        ff_out = self.mlp(x)

        x = self.norm3(x + self.dropout(ff_out))

        return x

class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_heads: int,
                 num_layers: int,
                 device: str,
                 d_ff: int,
                 dropout: float,
                 chunk_size: int,
                 activation_type = 'relu'
                 ):
        super(Decoder, self).__init__()
        self.emb_size = embedding_dim
        self.device = device
        self.chunk_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pe = nn.Embedding(chunk_size, embedding_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim_ff=d_ff,
                    num_heads=num_heads,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    activation_type=activation_type,

                )
                for _ in range(num_layers)

            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, padding_mask=None):
        # x: (bs, seq_len)
        #context: (bs, seq_len, emb_dim)
        N, seqlen, = x.shape

        out = self.chunk_emb(x)
        pos = torch.arange(0, seqlen).expand(N, seqlen).to(self.device)
        out = self.dropout(
            (out + self.pe(pos))
        )
        for layer in self.layers:
            out = layer(out, context, padding_mask=padding_mask)
        return out
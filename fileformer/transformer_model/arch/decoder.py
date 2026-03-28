import torch
import torch.nn as nn
from fileformer.transformer_model.arch.attention import MultiHeadLatentAttention
from fileformer.transformer_model.arch.pe import RotaryPositionalEmbeddings
from fileformer.transformer_model.arch.mlp import MLP
import torch.nn.functional as F
import math


class DecoderLayer(nn.Module):
    def __init__(self,
                 dim_ff: int,
                 num_heads: int,
                 embedding_dim: int,
                 activation_type: str,
                 dropout: float,
                 latent_dim:int,
                 ):
        super(DecoderLayer, self).__init__()
        # define attention
        self.head_dim = embedding_dim // num_heads
        self.self_attention = MultiHeadLatentAttention(embedding_dim, num_heads, latent_dim)
        #define mpl
        self.mlp = MLP(embedding_dim, dim_ff, activation_type, dropout)
        #define normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        self_attention_out = self.self_attention(x, x, x, mask=padding_mask)
        self_attention_out = self.dropout(self_attention_out)

        x = self.norm1(x + self_attention_out)

        ff_out = self.mlp(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x

class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_heads: int,
                 num_layers: int,
                 d_ff: int,
                 dropout: float,
                 latent_dim: int,
                 activation_type: str = 'relu',
                 ):
        super(Decoder, self).__init__()
        self.emb_size = embedding_dim
        self.chunk_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pe = RotaryPositionalEmbeddings(embedding_dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim_ff=d_ff,
                    num_heads=num_heads,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    activation_type=activation_type,
                    latent_dim=latent_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(embedding_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len):
        """
        Создает причинную маску для авторегрессии.
        Возвращает матрицу (seq_len, seq_len) где True означает "не смотреть" (запрещенные позиции).
        Для стандартного causal внимания: True в верхнем треугольнике (будущие позиции).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def forward(self, x, padding_mask=None):
        # x: (bs, seq_len)
        N, seq_len = x.shape
        device = x.device

        causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal = causal.view(1, 1, seq_len, seq_len)  # [B=1,H=1,L,L]

        if padding_mask is None:
            padding_mask = torch.zeros(N, seq_len, seq_len, dtype=torch.bool, device=device)

        pad = padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,L]
        pad = pad.expand(-1, -1, seq_len, -1)  # [B,1,L,L]

        combined = causal * (~pad)

        out = self.chunk_emb(x)
        out = self.pe(out)
        out = self.dropout(out)
        for layer in self.layers:
            out = layer(out, padding_mask=combined, )


        final_out = self.final_linear(out)

        return final_out
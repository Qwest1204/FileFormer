import torch
import torch.nn as nn
from fileformer.transformer_model.arch.attention import MultiHeadAttention, MultiHeadLatentAttention
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
                 latent_dim: int,
                 ):
        super(DecoderLayer, self).__init__()
        # define attention
        self.head_dim = embedding_dim // num_heads
        self.self_attention = MultiHeadLatentAttention(embedding_dim, num_heads, latent_dim)
        self.cross_attention = MultiHeadAttention(embedding_dim, num_heads, latent_dim)
        #define mpl
        self.mlp = MLP(embedding_dim, dim_ff, activation_type, dropout)
        #define normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, padding_mask=None, output_attentions=False):
        if output_attentions:
            self_attention_out, self_attn_weights = self.self_attention(x, x, x, mask=padding_mask,
                                                                        output_attentions=True)
            self_attention_out = self.dropout(self_attention_out)
        else:
            self_attention_out = self.self_attention(x, x, x, mask=padding_mask)
            self_attention_out = self.dropout(self_attention_out)

        x = self.norm1(x + self_attention_out)

        if output_attentions:
            cross_attention_out, cross_attn_weights = self.cross_attention(x, context, context, mask=None,
                                                                           output_attentions=True)
            cross_attention_out = self.dropout(cross_attention_out)
        else:
            cross_attention_out = self.cross_attention(x, context, context, mask=None)
            cross_attention_out = self.dropout(cross_attention_out)

        x = self.norm2(x + cross_attention_out)

        ff_out = self.mlp(x)
        x = self.norm3(x + self.dropout(ff_out))

        if output_attentions:
            # Возвращаем dict с weights для self и cross в этом слое
            return x, {"self_attn": self_attn_weights, "cross_attn": cross_attn_weights}
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
                 latent_dim: int,
                 activation_type: str = 'relu',
                 ):
        super(Decoder, self).__init__()
        self.emb_size = embedding_dim
        self.device = device
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
                    latent_dim=latent_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, context, padding_mask=None, output_attentions=False):
        # x: (bs, seq_len)
        #context: (bs, seq_len, emb_dim)
        N, seqlen = x.shape
        out = self.chunk_emb(x)
        pos = torch.arange(0, seqlen).expand(N, seqlen).to(self.device)
        out = self.dropout(
            (out + self.pe(out))
        )
        attentions = []  # Список dict'ов для каждого слоя
        for layer in self.layers:
            if output_attentions:
                out, layer_attns = layer(out, context, padding_mask=padding_mask, output_attentions=True)
                attentions.append(layer_attns)  # [{"self_attn": ..., "cross_attn": ...}, ...]
            else:
                out = layer(out, context, padding_mask=padding_mask)

        final_out = self.final_linear(out)

        if output_attentions:
            return final_out, attentions
        return final_out
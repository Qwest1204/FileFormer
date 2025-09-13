import torch
import torch.nn as nn
from notus.transformer_model.arch.attention import SelfAttention, MultiHeadAttention, MultiQueryAttention
from notus.transformer_model.arch.mlp import MLP

class EncoderBlock(nn.Module):
    def __init__(self,
                 dim_ff: int,
                 num_heads: int,
                 embedding_dim: int,
                 activation_type: str,
                 dropout: float,
                 ):
        super(EncoderBlock, self).__init__()
        # define attention
        self.head_dim = embedding_dim // num_heads
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        #define mpl
        self.mlp = MLP(embedding_dim, dim_ff, activation_type, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attention = self.attention(q, k, v, mask=mask)

        x = self.dropout(self.norm1(attention + q))
        forward = self.mlp(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 file_type_vocab: int,
                 num_heads: int,
                 num_layers: int,
                 device: str,
                 d_ff: int,
                 dropout: float,
                 hash_len: int,
                 activation_type = 'relu'
                 ):
        super(Encoder, self).__init__()
        self.emb_size = embedding_dim
        self.device = device
        self.file_type_emb = nn.Embedding(file_type_vocab, embedding_dim)
        self.hash_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pe = nn.Embedding(hash_len+1, embedding_dim)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    dim_ff=d_ff,
                    num_heads=num_heads,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    activation_type=activation_type,

                )
                for _ in range(num_layers)

            ]
        )

        self._init_weights()

        self.dropout = nn.Dropout(dropout)

    def forward(self, hash, type, mask=None):
        N, seqlen = hash.shape
        out = torch.cat([self.hash_emb(hash), self.file_type_emb(type)], dim=1)
        pos = torch.arange(0, seqlen+1).expand(N, seqlen+1).to(self.device)
        out = self.dropout(
            (out + self.pe(pos))
        )
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

    def _init_weights(self):
        # Инициализация эмбеддингов
        nn.init.xavier_uniform_(self.file_type_emb.weight)
        nn.init.xavier_uniform_(self.hash_emb.weight)
        nn.init.xavier_uniform_(self.pe.weight)

        # Инициализация для каждого слоя
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                if hasattr(layer.attention, 'w_q'):
                    nn.init.xavier_uniform_(layer.attention.w_q.weight)
                    nn.init.xavier_uniform_(layer.attention.w_k.weight)
                    nn.init.xavier_uniform_(layer.attention.w_v.weight)
                    nn.init.xavier_uniform_(layer.attention.w_o.weight)
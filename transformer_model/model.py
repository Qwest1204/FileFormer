import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert x.size(1) <= self.seq_len, "Sequence length exceeds the maximum allowed length."
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Reshape mask for proper broadcasting
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # Add dimensions for heads and query length
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add dimension for heads
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderCompress(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList, compress_factor: int = 2) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        self.compress_factor = compress_factor
        self.compress = nn.Sequential(
            nn.Conv1d(
                in_channels=features,
                out_channels=features,
                kernel_size=compress_factor,
                stride=compress_factor,
                padding=0
            ),
            nn.GELU()
        )
        self.res_compress = nn.Linear(features*compress_factor, features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        residual = x
        batch, seq_len, d_model = x.shape

        assert seq_len % self.compress_factor == 0, "Sequence length must be divisible by compress_factor."

        x = x.permute(0, 2, 1)
        x = self.compress(x)
        x = x.permute(0, 2, 1)

        residual = residual.view(batch, seq_len // self.compress_factor, d_model * self.compress_factor)
        residual = self.res_compress(residual)

        return x + residual

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class DecoderExpand(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList, expand_factor: int = 2) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        self.expand_factor = expand_factor
        self.expand = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=features,
                out_channels=features,
                kernel_size=expand_factor,
                stride=expand_factor,
                padding=0,
            ),
            nn.GELU()
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        batch, seq_len, d_model = x.shape
        x = x.permute(0, 2, 1)
        x = self.expand(x)
        x = x.permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, src_pos, projection_layer, compress):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer
        self.compress = compress

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask):
        if self.compress:
            compress_factor = self.encoder.compress_factor
            if src_mask is not None:
                if src_mask.dim() == 3:
                    compressed_mask = F.max_pool1d(
                        src_mask.float(),
                        kernel_size=compress_factor,
                        stride=compress_factor
                    )
                    src_mask = compressed_mask.to(src_mask.dtype)
                elif src_mask.dim() == 2:
                    compressed_mask = F.max_pool1d(
                        src_mask.unsqueeze(1).float(),
                        kernel_size=compress_factor,
                        stride=compress_factor
                    )
                    src_mask = compressed_mask.squeeze(1).to(src_mask.dtype)
                else:
                    raise RuntimeError(f"Unsupported src_mask dim: {src_mask.dim()}")
        return self.decoder(encoder_output, encoder_output, src_mask, None)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, src, src_mask):
        x = self.encode(src, src_mask)
        x = self.decode(x, src_mask)
        return self.project(x)

def build_transformer(vocab_size: int, d_model: int, max_seq_len: int, dropout: float, n_layers: int,
                      n_heads: int, d_ff: int, factor: int, compress: bool) -> Transformer:
    src_embed = InputEmbeddings(d_model, vocab_size)
    src_pos = PositionalEncoding(d_model, max_seq_len, dropout)

    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    if compress:
        encoder = EncoderCompress(d_model, nn.ModuleList(encoder_blocks), compress_factor=factor)
        decoder = DecoderExpand(d_model, nn.ModuleList(decoder_blocks), expand_factor=factor)
    else:
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, src_pos, projection_layer, compress)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

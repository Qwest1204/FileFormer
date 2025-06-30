import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class UpscaleNet(nn.Module):
    class ReshapeUpscale(nn.Module):
        def __init__(self, out_ch):
            super().__init__()
            self.out_ch = out_ch

        def forward(self, x):
            return x.view(x.size(0), x.size(1) * 2, self.out_ch)

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, 128 * 2),
            self.ReshapeUpscale(128),
            nn.GELU(),
            *self._make_upscale_block(128, 256),
            *self._make_upscale_block(256, 512),
            *self._make_upscale_block(512, 1024),
            *self._make_upscale_block(1024, 2048)
        )

    def _make_upscale_block(self, in_ch, out_ch):
        return [
            nn.Linear(in_ch, out_ch * 2),
            self.ReshapeUpscale(out_ch),
            nn.GELU()
        ]

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        x = x.permute(0, 2, 1)
        return x


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Динамическое расширение позиционного кодирования
            extra_len = seq_len - self.pe.size(1)
            extra_pe = torch.zeros(1, extra_len, self.d_model, device=x.device)
            position = torch.arange(self.pe.size(1), self.pe.size(1) + extra_len, dtype=torch.float,
                                    device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(
                x.device)
            extra_pe[0, :, 0::2] = torch.sin(position * div_term)
            extra_pe[0, :, 1::2] = torch.cos(position * div_term)
            self.pe = torch.cat([self.pe, extra_pe], dim=1)

        x = x + self.pe[:, :seq_len].detach()
        return self.dropout(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear2(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, window_size: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.window_size = window_size

        assert d_model % h == 0, "d_model must be divisible by h"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    @staticmethod
    def create_local_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        left_radius = (window_size - 1) // 2
        right_radius = window_size - 1 - left_radius

        row_idx = torch.arange(seq_len, device=device).view(-1, 1)
        col_idx = torch.arange(seq_len, device=device).view(1, -1)

        mask = (col_idx >= row_idx - left_radius) & (col_idx <= row_idx + right_radius)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len_q, _ = q.size()
        seq_len_k = k.size(1)

        # Linear projections
        query = self.w_q(q).view(batch_size, seq_len_q, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, seq_len_k, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, seq_len_k, self.h, self.d_k).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply local mask if applicable
        if self.window_size is not None and seq_len_q == seq_len_k:
            local_mask = self.create_local_mask(seq_len_q, self.window_size, attn_scores.device)
            attn_scores = attn_scores.masked_fill(~local_mask, -1e9)

        # Apply external mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self.attn_weights = attn_weights.detach()  # Save for visualization

        # Apply attention to values
        x = torch.matmul(attn_weights, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention: MultiHeadAttention,
                 feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(features, dropout)
        self.residual2 = ResidualConnection(features, dropout)

    def forward(self, x, src_mask):
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderCompress(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList, compress_factor: int = 2) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)
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
        self.res_compress = nn.Linear(features * compress_factor, features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        batch, seq_len, d_model = x.shape
        assert seq_len % self.compress_factor == 0, (
            f"Sequence length {seq_len} must be divisible by compress factor {self.compress_factor}"
        )

        # Compression
        x_compressed = x.permute(0, 2, 1)
        x_compressed = self.compress(x_compressed).permute(0, 2, 1)

        # Residual connection
        residual = x.reshape(batch, seq_len // self.compress_factor, d_model * self.compress_factor)
        residual = self.res_compress(residual)

        return x_compressed + residual


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention: MultiHeadAttention,
                 cross_attention: MultiHeadAttention, feed_forward: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(features, dropout)
        self.residual2 = ResidualConnection(features, dropout)
        self.residual3 = ResidualConnection(features, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class DecoderExpand(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList, expand_factor: int = 2) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)
        self.expand_factor = expand_factor
        self.expand = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=features,
                out_channels=features,
                kernel_size=expand_factor,
                stride=expand_factor,
                padding=0
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
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 src_embed: InputEmbeddings, src_pos: PositionalEncoding,
                 projection: ProjectionLayer, compress: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection = projection
        self.compress = compress

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor,
               src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Handle mask compression if needed
        if self.compress and src_mask is not None:
            compress_factor = self.encoder.compress_factor
            if src_mask.dim() == 2:
                src_mask = src_mask.unsqueeze(1)
                src_mask = F.max_pool1d(
                    src_mask.float(),
                    kernel_size=compress_factor,
                    stride=compress_factor
                ).squeeze(1).bool()
            elif src_mask.dim() == 3:
                src_mask = F.max_pool1d(
                    src_mask.float(),
                    kernel_size=compress_factor,
                    stride=compress_factor
                ).bool()

        return self.decoder(encoder_output, encoder_output, src_mask, None)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoded = self.encode(src, src_mask)
        decoded = self.decode(encoded, src_mask)
        return self.project(decoded)


class AttentionVisualizer:
    def __init__(self):
        self.attention_weights = {}
        self.hooks = []
        self.layer_names = []
        self.module_to_name = {}

    def _hook_fn(self, module, inputs, outputs):
        try:
            if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                attn = module.attn_weights
            else:
                return

            # Get module name
            if module not in self.module_to_name:
                layer_type = module.__class__.__name__
                layer_id = f"{layer_type}_{len(self.layer_names)}"
                self.module_to_name[module] = layer_id
                self.layer_names.append(layer_id)
            else:
                layer_id = self.module_to_name[module]

            # Store attention weights (detach and move to CPU)
            self.attention_weights[layer_id] = attn.detach().cpu()

        except Exception as e:
            print(f"Attention hook error: {e}")

    def register_hooks(self, model: nn.Module):
        self.module_to_name = {}
        for name, module in model.named_modules():
            if isinstance(module, MultiHeadAttention):
                self.module_to_name[module] = name
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
                print(f"Registered hook for: {name}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_attention(self, layer_id: str) -> Optional[torch.Tensor]:
        return self.attention_weights.get(layer_id)

    def visualize(self, layer_id: str, head: int = 0, save_path: str = None):
        if layer_id not in self.attention_weights:
            print(f"Layer {layer_id} not found. Available layers: {list(self.attention_weights.keys())}")
            return

        attn = self.attention_weights[layer_id]
        if head >= attn.shape[1]:
            print(f"Head index {head} out of range (max {attn.shape[1] - 1})")
            return

        plt.figure(figsize=(12, 10))
        plt.imshow(attn[0, head].numpy(), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f"Attention: {layer_id} - Head {head}")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved attention plot to {save_path}")
        plt.show()


def build_transformer(
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        compress_factor: int = 2,
        compress: bool = True,
        window_size: int = 64,
        return_attention: bool = False
) -> Transformer:
    # Embeddings and positional encoding
    src_embed = InputEmbeddings(d_model, vocab_size)
    src_pos = PositionalEncoding(d_model, max_seq_len, dropout)

    # Encoder blocks
    encoder_blocks = []
    for i in range(n_layers):
        self_attn = MultiHeadAttention(
            d_model=d_model,
            h=n_heads,
            dropout=dropout,
            window_size=window_size
        )
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, self_attn, ff_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder blocks
    decoder_blocks = []
    for i in range(n_layers):
        self_attn = MultiHeadAttention(
            d_model=d_model,
            h=n_heads,
            dropout=dropout,
            window_size=window_size  # Local window for self-attention
        )
        cross_attn = MultiHeadAttention(
            d_model=d_model,
            h=n_heads,
            dropout=dropout,
            window_size=None  # Global attention for cross-attention
        )
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, self_attn, cross_attn, ff_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    if compress:
        encoder = EncoderCompress(d_model, nn.ModuleList(encoder_blocks), compress_factor)
        decoder = DecoderExpand(d_model, nn.ModuleList(decoder_blocks), compress_factor)
    else:
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Projection layer
    projection = ProjectionLayer(d_model, vocab_size)

    # Create transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        src_pos=src_pos,
        projection=projection,
        compress=compress
    )

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Add attention visualization if requested
    if return_attention:
        transformer.attention_visualizer = AttentionVisualizer()
        transformer.attention_visualizer.register_hooks(transformer)

    return transformer
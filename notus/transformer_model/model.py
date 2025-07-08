import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

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
    def __init__(self, features: int, dmodel: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(dmodel)
        self.compress = nn.Sequential(
            nn.Linear(features, dmodel),
            nn.GELU()
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        batch, seq_len, d_model = x.shape

        # Compression
        x_compressed = x.permute(0, 2, 1)
        x_compressed = self.compress(x_compressed).permute(0, 2, 1)

        # Residual connection
        #residual = x.reshape(batch, seq_len // self.compress_factor, d_model * self.compress_factor)
        #residual = self.res_compress(residual)

        return x_compressed# + residual

from torch import Tensor


class Transformer(nn.Module):
    def __init__(self, encoder: nn.Module,
                 src_embed: nn.Module,
                 src_pos: nn.Module,
                 compress: bool,
                 output_dim: int = 2048) -> None:  # Добавляем параметр output_dim
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.compress = compress

        # Добавляем слои для нормализации и проекции
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_dim)  # Для фиксированного размера
        self.layer_norm = nn.LayerNorm(output_dim)  # Нормализация значений
        self.output_proj = nn.Linear(output_dim, output_dim)  # Дополнительная проекция

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        encoded = self.encoder(src, src_mask)

        # Транспонируем для работы с AdaptiveAvgPool1d
        # Исходная форма: [batch_size, seq_len, features]
        encoded = encoded.permute(0, 2, 1)  # Меняем местами seq_len и features

        # Приводим к фиксированному размеру
        pooled = self.adaptive_pool(encoded)  # [batch_size, features, output_dim]

        # Возвращаем к исходной размерности
        pooled = pooled.permute(0, 2, 1)  # [batch_size, output_dim, features]

        # Нормализация и проекция
        normalized = self.layer_norm(pooled)
        return self.output_proj(normalized)

    def decode(self, encoder_output: Tensor,
               src_mask: Optional[Tensor] = None) -> Tensor:
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

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None) -> Tensor:
        encoded = self.encode(src, src_mask)
        return encoded


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

class FileTransformer(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=6, max_seq_len=4096, embedding_tensor=1024, vocab_size=267):
        super().__init__()

        self.file_projection = nn.Linear(embedding_tensor, d_model)

        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, d_model))

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.out_proj = nn.Linear(d_model, 256)

    def forward(self, emb, x):
        file_vec = self.file_projection(emb.flatten(1))

        seq_emb = self.token_emb(x)
        seq_emb += self.pos_enc[:x.size(0), None, :]

        seq_emb += file_vec.unsqueeze(0)

        out = self.transformer(seq_emb)

        return self.out_proj(out)

def build_transformer(
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        compress: bool = True,
        window_size: int = 64,
        return_attention: bool = False,
        output_dim: int = 2048,
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

    # Create encoder and decoder
    if compress:
        encoder = EncoderCompress(max_seq_len, d_model, nn.ModuleList(encoder_blocks))
    else:
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Projection layer

    # Create transformer
    transformer = Transformer(
        encoder=encoder,
        src_embed=src_embed,
        src_pos=src_pos,
        compress=compress,
        output_dim=output_dim,
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class InputEmbeddings(nn.Module):
    """Converts input token IDs to embeddings and scales them by sqrt(d_model).

    Args:
        d_model: Dimension of embeddings
        vocab_size: Size of vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Adds positional encoding to input embeddings with dropout.

    Args:
        d_model: Dimension of embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len

        # Initialize positional encoding buffer
        self.register_buffer('pe', self._create_positional_encoding(max_len, d_model))

    @staticmethod
    def _create_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """Generate positional encoding tensor."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        # Dynamically extend positional encoding if needed
        if seq_len > self.pe.size(1):
            extra_len = seq_len - self.pe.size(1)
            extra_pe = self._create_positional_encoding(extra_len, self.d_model)
            self.pe = torch.cat([self.pe, extra_pe], dim=1)

        # Add positional encoding to input
        x = x + self.pe[:, :seq_len].detach()
        return self.dropout(x)


class FeedForwardBlock(nn.Module):
    """Position-wise feed-forward network with GELU activation and dropout.

    Args:
        d_model: Input/output dimension
        d_ff: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear2(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with optional local window attention.

    Args:
        d_model: Model dimension
        h: Number of attention heads
        dropout: Dropout probability
        window_size: Size of local attention window (None for global attention)
    """

    def __init__(self, d_model: int, h: int, dropout: float,
                 window_size: Optional[int] = None) -> None:
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
        self.attn_weights = None  # For attention visualization

    @staticmethod
    def create_local_mask(seq_len: int, window_size: int,
                          device: torch.device) -> torch.Tensor:
        """Create local attention mask with given window size."""
        left_radius = (window_size - 1) // 2
        right_radius = window_size - 1 - left_radius

        row_idx = torch.arange(seq_len, device=device).view(-1, 1)
        col_idx = torch.arange(seq_len, device=device).view(1, -1)

        mask = (col_idx >= row_idx - left_radius) & (col_idx <= row_idx + right_radius)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len_q, _ = q.size()
        seq_len_k = k.size(1)

        # Project and split into heads
        query = self.w_q(q).view(batch_size, seq_len_q, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, seq_len_k, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, seq_len_k, self.h, self.d_k).transpose(1, 2)

        # Calculate attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply local window mask if specified
        if self.window_size is not None and seq_len_q == seq_len_k:
            local_mask = self.create_local_mask(seq_len_q, self.window_size, attn_scores.device)
            attn_scores = attn_scores.masked_fill(~local_mask, -1e9)

        # Apply external mask (padding mask)
        if mask is not None:
            # Ensure mask has correct dimensions [batch, 1, 1, seq_len]
            while mask.dim() < 4:
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
    """Residual connection with layer normalization and dropout.

    Args:
        features: Feature dimension for layer norm
        dropout: Dropout probability
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """Apply residual connection: output = x + dropout(sublayer(norm(x)))"""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """Transformer encoder block with self-attention and feed-forward layers.

    Args:
        features: Feature dimension (d_model)
        self_attention: Self-attention module
        feed_forward: Feed-forward module
        dropout: Dropout probability
    """

    def __init__(self, features: int, self_attention: MultiHeadAttention,
                 feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(features, dropout)
        self.residual2 = ResidualConnection(features, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """Transformer encoder composed of multiple encoder blocks.

    Args:
        features: Feature dimension (d_model)
        layers: List of encoder blocks
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderCompress(nn.Module):
    """Encoder with sequence compression at the output.

    Args:
        features: Input feature dimension
        dmodel: Output feature dimension
        layers: List of encoder blocks
    """

    def __init__(self, features: int, dmodel: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(dmodel)
        self.compress = nn.Sequential(
            nn.Linear(features, dmodel),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Process through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        # Compress sequence dimension
        x_compressed = x.permute(0, 2, 1)
        x_compressed = self.compress(x_compressed).permute(0, 2, 1)
        return x_compressed


class Transformer(nn.Module):
    """Transformer model with encoder-only architecture.

    Args:
        encoder: Encoder module
        src_embed: Source embedding module
        src_pos: Positional encoding module
        compress: Whether to use sequence compression
    """

    def __init__(self, encoder: nn.Module,
                 src_embed: nn.Module,
                 src_pos: nn.Module,
                 compress: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.compress = compress

    def encode(self, src: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input sequence."""
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer."""
        return self.encode(src, src_mask)


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
) -> Transformer:
    """Construct a Transformer model with specified hyperparameters.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        n_layers: Number of encoder layers
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        compress: Use sequence compression
        window_size: Local attention window size
        return_attention: Enable attention visualization

    Returns:
        Configured Transformer model
    """
    # Embeddings and positional encoding
    src_embed = InputEmbeddings(d_model, vocab_size)
    src_pos = PositionalEncoding(d_model, max_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(n_layers):
        self_attn = MultiHeadAttention(
            d_model=d_model,
            h=n_heads,
            dropout=dropout,
            window_size=window_size
        )
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, self_attn, ff_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create encoder
    if compress:
        encoder = EncoderCompress(max_seq_len, d_model, nn.ModuleList(encoder_blocks))
    else:
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Create transformer
    transformer = Transformer(
        encoder=encoder,
        src_embed=src_embed,
        src_pos=src_pos,
        compress=compress,
    )

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
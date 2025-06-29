import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from typing import Dict, List
from typing import Optional

class UpscaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Последовательность слоёв
        self.layers = nn.Sequential(
            # Этап 1: Увеличение каналов (16 -> 64)
            nn.Conv1d(16, 64, kernel_size=3, padding=1),
            nn.GELU(),

            # Этап 2: Удвоение длины (32 -> 64)
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            # Повторить 5 раз (64 -> 128 -> 256 -> 512 -> 1024 -> 2048)
            *self._make_upscale_block(128, 256),  # 64 -> 128
            *self._make_upscale_block(256, 512),  # 128 -> 256
            *self._make_upscale_block(512, 1024), # 256 -> 512
            #*self._make_upscale_block(1024, 2048), # 512 -> 1024
            #*self._make_upscale_block(2048, 2048)  # 1024 -> 2048 (каналы)
        )

    def _make_upscale_block(self, in_ch, out_ch):
        return [
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ]

    def forward(self, x):
        return self.layers(x)

class LayerNormalization(nn.Module):
    """
    Layer normalization layer that normalizes the input features.
    """
    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        Forward pass. Computes mean and standard deviation, then normalizes the input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Feed-forward block with two linear layers and dropout.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass. Applies first linear layer, ReLU activation, dropout, and second linear layer.
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    """
    Input embedding layer that maps token indices to dense vectors.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass. Embeds input tokens and scales by sqrt(d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings using sine and cosine functions.
    """
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
        """
        Forward pass. Adds positional encoding to the input and applies dropout.
        """
        assert x.size(1) <= self.seq_len, "Sequence length exceeds the maximum allowed length."
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization and dropout.
    """
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """
        Forward pass. Applies layer normalization, sublayer, dropout, and adds residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention block that splits the input into multiple heads for parallel attention computation.
    """
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
        """
        Computes the attention scores and applies them to the value.
        """
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
        """
        Forward pass. Computes query, key, and value matrices, splits them into heads, computes attention, and concatenates the heads.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        self.attn_weights = self.attention_scores.detach()
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)





class LocalAttentionBlock(nn.Module):
    """
    Multi-head attention with local attention window.
    Inherits from MultiHeadAttentionBlock and adds local window constraint.
    """

    def __init__(self, d_model: int, h: int, dropout: float, window_size: Optional[int] = None) -> None:
        """
        Args:
            d_model: Dimension of the model.
            h: Number of attention heads.
            dropout: Dropout probability.
            window_size: Size of the local attention window. If None, uses global attention.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        assert d_model % h == 0, "d_model is not divisible by h"

        # Linear transformations for query, key, value, and output
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size  # Size of the local attention window

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        """
        Forward pass with local attention.

        Args:
            q: Query tensor (batch_size, seq_len, d_model)
            k: Key tensor (batch_size, seq_len, d_model)
            v: Value tensor (batch_size, seq_len, d_model)
            mask: Optional mask tensor (batch_size, seq_len) or (batch_size, seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        # Apply linear transformations and split into heads
        query = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply local window constraint only for self-attention (when seq_len_q == seq_len_k)
        if self.window_size is not None and seq_len_q == seq_len_k:
            local_mask = self.create_local_mask(seq_len_q, self.window_size, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(local_mask == 0, -1e9)

        # Apply external mask (e.g., padding mask or future mask)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Normalize and apply dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self.attn_weights = attn_weights.detach()
        # Apply attention weights to values
        x = torch.matmul(attn_weights, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)

    @staticmethod
    def create_local_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """
        Creates a mask for local attention.

        Args:
            seq_len: Length of the sequence.
            window_size: Size of the local attention window.
            device: Device to place the mask on.

        Returns:
            Mask tensor (1, 1, seq_len, seq_len) where 1 indicates allowed positions.
        """
        # Calculate left and right radii for asymmetric windows near boundaries
        left_radius = (window_size - 1) // 2
        right_radius = window_size - 1 - left_radius

        # Create indices for the sequence
        row_indices = torch.arange(seq_len, device=device).view(-1, 1)  # (seq_len, 1)
        col_indices = torch.arange(seq_len, device=device).view(1, -1)  # (1, seq_len)

        # Compute distance matrix
        distance = torch.abs(col_indices - row_indices)

        # Create mask where only positions within window are allowed
        mask = (distance <= left_radius) | (distance <= right_radius)
        mask = mask.float()

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
class EncoderBlock(nn.Module):
    """
    Encoder block consisting of self-attention and feed-forward layers with residual connections.
    """
    def __init__(self, features: int, self_attention_block: LocalAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Forward pass. Applies self-attention and feed-forward layers with residual connections.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    """
    Encoder consisting of a stack of encoder blocks and layer normalization.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """
        Forward pass. Applies each encoder block in sequence and normalizes the output.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderCompress(nn.Module):
    """
    Encoder with compression mechanism that reduces sequence length by a factor.
    """
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
        """
        Forward pass. Applies encoder blocks, compresses the sequence, and adds residual connection.
        """
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
    """
    Decoder block consisting of self-attention, cross-attention, and feed-forward layers with residual connections.
    """
    def __init__(self, features: int, self_attention_block: LocalAttentionBlock,
                 cross_attention_block: LocalAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass. Applies self-attention, cross-attention, and feed-forward layers with residual connections.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """
    Decoder consisting of a stack of decoder blocks and layer normalization.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass. Applies each decoder block in sequence and normalizes the output.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class DecoderExpand(nn.Module):
    """
    Decoder with expansion mechanism that increases sequence length by a factor.
    """
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
        """
        Forward pass. Expands the sequence length and applies decoder blocks.
        """
        batch, seq_len, d_model = x.shape
        x = x.permute(0, 2, 1)
        x = self.expand(x)
        x = x.permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Projection layer that maps decoder output to vocabulary size.
    """
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass. Projects input to vocabulary size.
        """
        return self.proj(x)

class Transformer(nn.Module):
    """
    Transformer model consisting of encoder, decoder, embeddings, positional encoding, and projection layer.
    """
    def __init__(self, encoder, decoder, src_embed, src_pos, projection_layer, compress):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer
        self.compress = compress

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask):
        """
        Decodes the encoder output.
        """
        if self.compress:
            compress_factor = self.encoder.compress_factor
            if src_mask is not None:
                # Для 2D масок [batch, seq_len]
                if src_mask.dim() == 2:
                    src_mask = src_mask.view(src_mask.size(0), 1, src_mask.size(1))
                    src_mask = F.max_pool1d(
                        src_mask.float(),
                        kernel_size=compress_factor,
                        stride=compress_factor
                    ).squeeze(1).to(torch.long)

                # Для 3D масок [batch, 1, seq_len]
                elif src_mask.dim() == 3:
                    src_mask = F.max_pool1d(
                        src_mask.float(),
                        kernel_size=compress_factor,
                        stride=compress_factor
                    ).to(torch.long)
                else:
                    raise RuntimeError(f"Unsupported src_mask dim: {src_mask.dim()}")
        return self.decoder(encoder_output, encoder_output, src_mask, None)

    def project(self, x):
        """
        Projects the decoder output to vocabulary size.
        """
        return self.projection_layer(x)

    def forward(self, src, src_mask):
        """
        Forward pass. Encodes the source, decodes the encoded output, and projects to vocabulary size.
        """
        x = self.encode(src, src_mask)
        x = self.decode(x, src_mask)
        return self.project(x)


class AttentionVisualizer:
    def __init__(self):
        self.attention_weights = {}
        self.hooks = []
        self.layer_names = []
        self.module_to_name = {}  # Сопоставление модулей с именами

    def _hook_fn(self, module, input, output):
        try:
            attn_weights = None

            # Пытаемся получить веса внимания разными способами
            if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                attn_weights = module.attn_weights
            elif hasattr(module, 'attention_scores') and module.attention_scores is not None:
                attn_weights = module.attention_scores
            elif isinstance(output, tuple) and len(output) == 2:
                attn_weights = output[1]

            if attn_weights is None:
                return
            print(f"Хук сработал для: {module}, weights found: {attn_weights is not None}")
            # Используем имя модуля из сопоставления
            if module in self.module_to_name:
                layer_id = self.module_to_name[module]
            else:
                layer_type = module.__class__.__name__
                layer_id = f"{layer_type}_{len(self.layer_names)}"
                self.module_to_name[module] = layer_id
                self.layer_names.append(layer_id)

            # Сохраняем веса
            self.attention_weights[layer_id] = attn_weights.detach().cpu()

        except Exception as e:
            print(f"Ошибка в хуке: {e}")

    def register_hooks(self, model: nn.Module):
        """Регистрация хуков с сохранением имён модулей"""
        self.module_to_name = {}
        for name, module in model.named_modules():
            if isinstance(module, (MultiHeadAttentionBlock, LocalAttentionBlock)):
                self.module_to_name[module] = name
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
                print(f"Зарегистрирован хук для: {name}")


    def print_available_layers(self):
        """Печать всех доступных слоёв с весами внимания"""
        print("\nДоступные слои для визуализации:")
        for i, name in enumerate(self.layer_names):
            print(f"{i}: {name}")

    def visualize(self, layer_id, head: int = 0, save_path: str = None):
        """
        Визуализация весов внимания для выбранного слоя и головы

        Args:
            layer_id: ID слоя (число или имя)
            head: Номер головы для отображения
        """
        # Преобразуем числовой ID в имя
        if isinstance(layer_id, int):
            layer_id = self.layer_names[layer_id] if layer_id < len(self.layer_names) else None

        if layer_id not in self.attention_weights:
            print(f"Слой {layer_id} не найден. Доступные слои:")
            self.print_available_layers()
            return

        attn = self.attention_weights[layer_id]
        print(f"Форма весов внимания для {layer_id}: {attn.shape}")

        # Выбор батча 0 и указанной головы
        attn_matrix = attn[0, head]

        plt.figure(figsize=(10, 8))
        plt.imshow(attn_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Attention: {layer_id} - Head {head}")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

def build_transformer(vocab_size: int, d_model: int, max_seq_len: int, dropout: float, n_layers: int,
                      n_heads: int, d_ff: int, factor: int, compress: bool, winsize: int, return_attention: bool) -> Transformer:
    """
    Builds a transformer model with specified parameters.
    """
    src_embed = InputEmbeddings(d_model, vocab_size)
    src_pos = PositionalEncoding(d_model, max_seq_len, dropout)
    # Initializes input embeddings and positional encoding.

    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = LocalAttentionBlock(window_size=winsize, d_model=d_model, h=n_heads, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    # Creates encoder blocks with self-attention and feed-forward layers.

    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attention_block = LocalAttentionBlock(window_size=winsize, d_model=d_model, h=n_heads, dropout=dropout)
        decoder_cross_attention_block = LocalAttentionBlock(window_size=winsize, d_model=d_model, h=n_heads, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    # Creates decoder blocks with self-attention, cross-attention, and feed-forward layers.

    if compress:
        encoder = EncoderCompress(d_model, nn.ModuleList(encoder_blocks), compress_factor=factor)
        decoder = DecoderExpand(d_model, nn.ModuleList(decoder_blocks), expand_factor=factor)
    else:
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    # Initializes encoder and decoder with or without compression/expansion based on the compress flag.

    projection_layer = ProjectionLayer(d_model, vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, src_pos, projection_layer, compress)
    # Initializes projection layer and creates transformer model.

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # Initializes model parameters with Xavier uniform initialization.

    if return_attention:
        # Создаем атрибут для сбора внимания
        transformer.attention_visualizer = AttentionVisualizer()
        transformer.attention_visualizer.register_hooks(transformer)

    return transformer
    # Returns the constructed transformer model.

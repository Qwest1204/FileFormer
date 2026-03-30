import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention.

    Implements the core attention mechanism without multi-head splitting.
    Q, K, V are projected to full `embedding_dim` (not split by heads).
    """

    def __init__(self, embedding_dim: int, head_dim: int):
        """Initialize SelfAttention.

        Args:
            embedding_dim (int): Input/output embedding dimension.
            head_dim (int): Dimension used for scaling factor (usually equals `embedding_dim`).
        """
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim

        self.Q_layer = nn.Linear(embedding_dim, embedding_dim)
        self.K_layer = nn.Linear(embedding_dim, embedding_dim)
        self.V_layer = nn.Linear(embedding_dim, embedding_dim)
        self.out_fc = nn.Linear(embedding_dim, embedding_dim)
        self.scale_param = head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        """Forward pass.

        Args:
            q (torch.Tensor): Query tensor (bs, seq_len, embedding_dim).
            k (torch.Tensor): Key tensor (bs, seq_len, embedding_dim).
            v (torch.Tensor): Value tensor (bs, seq_len, embedding_dim).
            mask (torch.Tensor, optional): Attention mask (bs, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (bs, seq_len, embedding_dim).
        """
        bs, seqlen, dim = q.shape

        Q = self.Q_layer(q)     #[bs, seq_len, head_dim]
        K = self.K_layer(k)     #[bs, seq_len, head_dim]
        V = self.V_layer(v)     #[bs, seq_len, head_dim]

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale_param #[bs, seq_len, seq_len]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        return self.out_fc(output)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        qkv_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(emb_size, emb_size, bias=qkv_bias)
        self.k_proj = nn.Linear(emb_size, emb_size, bias=qkv_bias)
        self.v_proj = nn.Linear(emb_size, emb_size, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_size, emb_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        mask: torch.Tensor = None,
        is_causal: bool = True,
    ):
        """
        Args:
            query: (batch, seq_len_q, emb_size)
            key:   (batch, seq_len_kv, emb_size). Если None, используется query.
            value: (batch, seq_len_kv, emb_size). Если None, используется key.
            mask:  Булева маска (True = игнорировать). Может быть:
                   - (batch, seq_len_kv)
                   - (batch, seq_len_q, seq_len_kv)
                   - (batch, num_heads, seq_len_q, seq_len_kv)
            is_causal: Флаг причинной маски (только для self-attention, когда q_len == kv_len).

        Returns:
            out: (batch, seq_len_q, emb_size)
            Если return_weights=True, дополнительно возвращает веса внимания (batch, num_heads, seq_len_q, seq_len_kv)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        bs, q_len, _ = query.shape
        kv_len = key.shape[1]

        # Проекции
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Разделение на головы
        Q = Q.view(bs, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, q_len, head_dim)
        K = K.view(bs, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bs, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Если нужны веса – используем ручной расчёт (медленнее, но даёт веса)
        # --- Используем SDPA (оптимизированный, с FlashAttention если доступно) ---
        # Подготовка маски для SDPA: маска должна быть булевой (True = attend) или числовой
        attn_mask = None
        if is_causal and mask is not None:
            # Объединяем каузальную маску с маской паддингов
            # Создаём каузальную маску (True = игнорировать)
            causal = torch.triu(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=Q.device), diagonal=1
            )  # (q_len, kv_len)
            causal = causal[None, None, :, :]  # (1, 1, q_len, kv_len)

            # Приводим переданную маску к булевому формату (True = игнорировать)
            if mask.dtype != torch.bool:
                # Если маска числовая, конвертируем: 0 -> attend, ненулевое -> ignore
                mask = mask != 0
            # Расширяем размерности маски до (batch, 1, q_len, kv_len) для broadcast
            if mask.dim() == 2:  # (bs, kv_len)
                mask = mask[:, None, None, :]  # (bs, 1, 1, kv_len)
            elif mask.dim() == 3:  # (bs, q_len, kv_len)
                mask = mask[:, None, :, :]    # (bs, 1, q_len, kv_len)
            # Объединяем: игнорировать, если хотя бы одна маска True
            combined_mask = causal | mask  # broadcast: (1,1,q_len,kv_len) + (bs,1,q_len,kv_len) -> (bs,1,q_len,kv_len)
            # Для SDPA нужна маска, где True = attend, поэтому инвертируем
            attn_mask = ~combined_mask
            is_causal = False  # явная маска передаётся
        elif is_causal:
            # Только причинность, паддингов нет
            attn_mask = None
            # is_causal остаётся True
        else:
            # Только паддинги (без причинности)
            if mask is not None:
                if mask.dtype == torch.bool:
                    attn_mask = ~mask  # инвертируем (True=attend)
                else:
                    # Если маска числовая, используем как есть (будет добавлена к scores)
                    attn_mask = mask
            # is_causal уже False

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(bs, q_len, self.emb_size)
        out = self.out_proj(out)
        return out
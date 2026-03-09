import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Standard multi-head scaled dot-product attention.

    Supports cross-attention (different seq lengths for Q vs K/V) and optional attention weight return.
    """

    def __init__(self, emb_size: int, num_heads: int, latent_dim: int):
        """Initialize MultiHeadAttention.

        Args:
            emb_size (int): Model embedding dimension.
            num_heads (int): Number of attention heads (must divide `emb_size`).
            latent_dim (int): Unused legacy parameter (kept for API compatibility).
        """
        super(MultiHeadAttention, self).__init__()
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.emb_size = emb_size

        self.Q_layer = nn.Linear(emb_size, emb_size)
        self.K_layer = nn.Linear(emb_size, emb_size)
        self.V_layer = nn.Linear(emb_size, emb_size)

        self.fc_out = nn.Linear(emb_size, emb_size)
        self.scale_param = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None, output_attentions=False):
        """Forward pass.

        Args:
            q (torch.Tensor): Query (bs, seq_len_q, emb_size).
            k (torch.Tensor): Key (bs, seq_len_kv, emb_size).
            v (torch.Tensor): Value (bs, seq_len_kv, emb_size).
            mask (torch.Tensor, optional): Attention mask.
            output_attentions (bool): If True, also return attention weights.

        Returns:
            torch.Tensor or tuple: Output (bs, seq_len_q, emb_size) and optional weights.
        """
        bs, seqlen_q, dim = q.shape
        _, seqlen_kv, _ = k.shape

        Q = self.Q_layer(q)  # (bs, seqlen_q, dim)
        K = self.K_layer(k)  # (bs, seqlen_kv, dim)
        V = self.V_layer(v)  # (bs, seqlen_kv, dim)

        # Reshape Q, K, V to (bs, num_heads, seq_len, head_dim)
        Q = Q.view(bs, seqlen_q, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seqlen_q, head_dim)
        K = K.view(bs, seqlen_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seqlen_kv, head_dim)
        V = V.view(bs, seqlen_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seqlen_kv, head_dim)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale_param

        if mask is not None:
            # Ensure mask is compatible with attention_scores
            if mask.dim() == 3:  # Assume mask is (bs, seqlen_q, seqlen_kv)
                mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # (bs, num_heads, seqlen_q, seqlen_kv)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(bs, seqlen_q, dim)
        output = self.fc_out(output)

        if output_attentions:
            return output, attention_weights
        return output


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA).

    Multiple query heads share single key/value head (memory-efficient for decoding).
    """

    def __init__(self, emb_size: int, num_heads: int, latent_dim: int):
        """Initialize MultiQueryAttention.

        Args:
            emb_size (int): Model embedding dimension.
            num_heads (int): Number of query heads (must divide `emb_size`).
            latent_dim (int): Unused legacy parameter.
        """
        super(MultiQueryAttention, self).__init__()
        assert emb_size % num_heads == 0, "Emb_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.Q_layer = nn.Linear(emb_size, num_heads * self.head_dim)
        self.K_layer = nn.Linear(emb_size, self.head_dim)
        self.V_layer = nn.Linear(emb_size, self.head_dim)

        self.fc_out = nn.Linear(num_heads * self.head_dim, emb_size)
        self.scale_param = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        """Forward pass.

        Args:
            q (torch.Tensor): Query (bs, seq_len, emb_size).
            k (torch.Tensor): Key (bs, seq_len_kv, emb_size).
            v (torch.Tensor): Value (bs, seq_len_kv, emb_size).
            mask (torch.Tensor, optional): Attention mask.

        Returns:
            torch.Tensor: Output (bs, seq_len, emb_size).
        """
        bs, seqlen, dim = q.shape
        _, seq_len_k, _ = k.shape

        Q = self.Q_layer(q) # (bs, seq_len, num_heads*head_dim)
        K = self.K_layer(k) # (bs, seq_len, head_dim)
        V = self.V_layer(v) # (bs, seq_len, head_dim)

        Q = Q.view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        K_t = K.unsqueeze(1).transpose(-1, -2)
        V_exp = V.unsqueeze(1)

        attention_scores = torch.matmul(Q, K_t) * self.scale_param
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V_exp)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, self.num_heads * self.head_dim)
        return self.fc_out(output)


class MultiHeadLinearAttention(nn.Module):
    """Multi-head linear attention (kernelized approximation of softmax attention).

    Uses feature map φ(x) = ELU(x) + 1 for linear-time complexity.
    Supports causal and non-causal modes.
    """

    def __init__(self, emb_size: int, num_heads: int, latent_dim=None):
        """Initialize MultiHeadLinearAttention.

        Args:
            emb_size (int): Model embedding dimension.
            num_heads (int): Number of heads (must divide `emb_size`).
            latent_dim (int, optional): Unused legacy parameter.
        """
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else self.head_dim
        self.Q_layer = nn.Linear(emb_size, num_heads * self.head_dim)
        self.K_layer = nn.Linear(emb_size, num_heads * self.head_dim)
        self.V_layer = nn.Linear(emb_size, num_heads * self.head_dim)
        self.fc_out = nn.Linear(num_heads * self.head_dim, emb_size)
        self.scale_param = self.head_dim ** -0.5

    def _reshape_to_heads(self, x):
        B, N, _ = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        return x.view(B * self.num_heads, N, self.head_dim)

    def _reshape_from_heads(self, x):
        B = x.shape[0] // self.num_heads
        x = x.view(B, self.num_heads, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
        return x.view(B, -1, self.num_heads * self.head_dim)

    def phi(self, x):
        """Feature map for linear attention."""
        return F.elu(x) + 1

    def forward(self, query, key, value, mask=None, causal=False):
        """Forward pass.

        Args:
            query (torch.Tensor): Query (B, N, emb_size).
            key (torch.Tensor): Key (B, N, emb_size).
            value (torch.Tensor): Value (B, N, emb_size).
            mask (torch.Tensor, optional): Padding mask.
            causal (bool): Whether to use causal (autoregressive) mode.

        Returns:
            torch.Tensor: Output (B, N, emb_size).
        """
        B = query.shape[0]
        q = self.Q_layer(query)
        k = self.K_layer(key)
        v = self.V_layer(value)

        q = self._reshape_to_heads(q)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)

        q = q * self.scale_param
        k = k * self.scale_param

        phi_q = self.phi(q)
        phi_k = self.phi(k)

        if mask is not None:
            pad_mask = mask.repeat_interleave(self.num_heads, dim=0).unsqueeze(-1)
            phi_k = phi_k.masked_fill(pad_mask, 0.)
            v = v.masked_fill(pad_mask, 0.)

        if causal:
            Z_cum = torch.cumsum(phi_k, dim=1)
            S_cum = torch.cumsum(phi_k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
            num = torch.matmul(phi_q.unsqueeze(-2), S_cum).squeeze(-2)
            den = (phi_q * Z_cum).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_output = num / den
        else:
            S = torch.matmul(phi_k.transpose(1, 2), v)
            Z = phi_k.sum(dim=1, keepdim=True)
            num = torch.matmul(phi_q, S)
            den = torch.matmul(phi_q, Z.transpose(-1, -2)).clamp(min=1e-8)
            attn_output = num / den

        attn_output = self._reshape_from_heads(attn_output)
        return self.fc_out(attn_output)


class MultiHeadLatentAttention(nn.Module):
    """Multi-head attention with latent (compressed) key/value dimension.

    Projects queries/keys/values into a smaller latent space per head for efficiency.
    """

    def __init__(self, emb_size: int, num_heads: int, latent_dim: int):
        super(MultiHeadLatentAttention, self).__init__()
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.latent_dim = latent_dim

        self.Q_to_latent = nn.Linear(emb_size, num_heads * latent_dim)
        self.K_to_latent = nn.Linear(emb_size, num_heads * latent_dim)
        self.V_to_latent = nn.Linear(emb_size, num_heads * latent_dim)

        self.fc_out = nn.Linear(num_heads * latent_dim, emb_size)
        self.scale_param = self.latent_dim ** -0.5

    def _reshape_to_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.latent_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size * self.num_heads, seq_len, self.latent_dim)

    def _reshape_from_heads(self, x):
        batch_size = x.shape[0] // self.num_heads
        x = x.view(batch_size, self.num_heads, -1, self.latent_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.num_heads * self.latent_dim)

    def forward(self, query, key, value, mask=None, causal=False):
        """Forward pass with improved masking.

        Args:
            query: (bs, seqlen_q, emb_size)
            key:   (bs, seqlen_k, emb_size)
            value: (bs, seqlen_k, emb_size)
            mask:  Padding mask of shape (bs, seqlen_k) with 0 for padding positions.
                   Positions with 0 will be masked out.
            causal: If True, applies a causal mask where query i cannot attend to key j if j > i.
                    For correct behaviour, seqlen_q should equal seqlen_k. For decoding with
                    kv cache, set causal=False and provide an appropriate mask.

        Returns:
            torch.Tensor: (bs, seqlen_q, emb_size)
        """
        bs, seqlen_q, _ = query.shape
        _, seqlen_k, _ = key.shape

        # Project to latent space
        q = self.Q_to_latent(query)  # (bs, seqlen_q, num_heads * latent_dim)
        k = self.K_to_latent(key)  # (bs, seqlen_k, num_heads * latent_dim)
        v = self.V_to_latent(value)  # (bs, seqlen_k, num_heads * latent_dim)

        # Reshape to separate heads
        q = self._reshape_to_heads(q)  # (bs * num_heads, seqlen_q, latent_dim)
        k = self._reshape_to_heads(k)  # (bs * num_heads, seqlen_k, latent_dim)
        v = self._reshape_to_heads(v)  # (bs * num_heads, seqlen_k, latent_dim)

        # Compute attention scores
        attn_scores = torch.einsum('bnd,bmd->bnm', q, k) * self.scale_param
        # shape: (bs * num_heads, seqlen_q, seqlen_k)

        # --- Causal mask ---
        if causal:
            # Build causal mask (True = mask out future positions)
            causal_mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, device=attn_scores.device, dtype=torch.bool),
                diagonal=1
            )  # (seqlen_q, seqlen_k)
            # Expand to all heads and batch
            causal_mask = causal_mask.unsqueeze(0).expand(bs * self.num_heads, seqlen_q, seqlen_k)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # --- Padding mask ---
        if mask is not None:
            # mask: (bs, seqlen_k) with 0 for padding positions
            pad_mask = (mask == 0)  # True where padding
            pad_mask = pad_mask.unsqueeze(1)  # (bs, 1, seqlen_k) – broadcast over seqlen_q
            # Repeat over heads
            pad_mask = pad_mask.repeat_interleave(self.num_heads, dim=0)  # (bs*heads, 1, seqlen_k)
            attn_scores = attn_scores.masked_fill(pad_mask, float('-inf'))

        # Softmax and output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_output = torch.einsum('bnm,bmd->bnd', attn_weights, v)  # (bs*heads, seqlen_q, latent_dim)
        # Merge heads and project back
        attn_output = self._reshape_from_heads(attn_output)  # (bs, seqlen_q, num_heads * latent_dim)
        attn_output = self.fc_out(attn_output)  # (bs, seqlen_q, emb_size)

        return attn_output
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim:int, head_dim:int):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim

        self.Q_layer = nn.Linear(embedding_dim, embedding_dim)
        self.K_layer = nn.Linear(embedding_dim, embedding_dim)
        self.V_layer = nn.Linear(embedding_dim, embedding_dim)
        self.out_fc = nn.Linear(embedding_dim, embedding_dim)
        self.scale_param = head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
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
    def __init__(self, emb_size, num_heads, latent_dim):
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
        bs, seqlen_q, dim = q.shape  # Extract seqlen from q for Q
        _, seqlen_kv, _ = k.shape

        Q = self.Q_layer(q)  # (bs, seqlen_q, dim)
        K = self.K_layer(k)  # (bs, seqlen_kv, dim)
        V = self.V_layer(v)  # (bs, seqlen_kv, dim)

        # Reshape Q, K, V to (bs, num_heads, seq_len, head_dim)
        Q = Q.view(bs, seqlen_q, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seqlen_q, head_dim)
        K = K.view(bs, seqlen_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seqlen_kv, head_dim)
        V = V.view(bs, seqlen_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seqlen_kv, head_dim)

        attention_scores = torch.matmul(Q,
                                        K.transpose(-1, -2)) * self.scale_param  # (bs, num_heads, seqlen_q, seqlen_kv)

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
    def __init__(self, emb_size: int, num_heads: int, latent_dim):
        super(MultiQueryAttention, self).__init__()
        assert emb_size % num_heads == 0, "Emb_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.Q_layer = nn.Linear(emb_size, num_heads* self.head_dim)
        self.K_layer = nn.Linear(emb_size, self.head_dim)
        self.V_layer = nn.Linear(emb_size, self.head_dim)

        self.fc_out = nn.Linear(num_heads*self.head_dim, emb_size)
        self.scale_param = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        bs, seqlen, dim = q.shape
        _, seq_len_k, _ = k.shape

        Q = self.Q_layer(q) # (bs, seq_len, num_heads*head_dim)
        K = self.K_layer(k) # (bs, seq_len, head_dim)
        V = self.V_layer(v) # (bs, seq_len, head_dim)

        Q = Q.view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2) # (bs, seq_len, num_heads*head_dim) -> (bs, num_heads, seq_len, head_dim)
        K_t = K.unsqueeze(1).transpose(-1, -2) # (bs, seq_len, head_dim) -> (bs, 1, head_dim, seq_len_k)
        V_exp = V.unsqueeze(1)

        attention_scores = torch.matmul(Q, K_t) * self.scale_param #[bs, num_heads, seq_len_q, seq_len_k]
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1) # (bs, num_heads, seq_len_q, seq_len_k)
        output = torch.matmul(attention_weights, V_exp)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, self.num_heads * self.head_dim)
        return self.fc_out(output)

class MultiHeadLinearAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, latent_dim=None):
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
        return F.elu(x) + 1

    def forward(self, query, key, value, mask=None, causal=False):
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
    def __init__(self, emb_size: int, num_heads: int, latent_dim: int):
        super(MultiHeadLatentAttention, self).__init__()
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.latent_dim = latent_dim  # Размерность латентного пространства

        # Проекции в латентное пространство
        self.Q_to_latent = nn.Linear(emb_size, num_heads * latent_dim)
        self.K_to_latent = nn.Linear(emb_size, num_heads * latent_dim)
        self.V_to_latent = nn.Linear(emb_size, num_heads * latent_dim)

        # Выходной слой
        self.fc_out = nn.Linear(num_heads * latent_dim, emb_size)
        self.scale_param = self.latent_dim ** -0.5

    def _reshape_to_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.latent_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, H, N, latent_dim)
        return x.view(batch_size * self.num_heads, seq_len, self.latent_dim)

    def _reshape_from_heads(self, x):
        batch_size = x.shape[0] // self.num_heads
        x = x.view(batch_size, self.num_heads, -1, self.latent_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, H, latent_dim)
        return x.view(batch_size, -1, self.num_heads * self.latent_dim)

    def forward(self, query, key, value, mask=None, output_attentions=False):
        bs, seqlen, dim = query.shape

        q = self.Q_to_latent(query)  # (bs, seqlen, num_heads * latent_dim)
        k = self.K_to_latent(key)
        v = self.V_to_latent(value)

        q = self._reshape_to_heads(q)  # (bs*num_heads, seqlen, latent_dim)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)

        attention_scores = torch.einsum('bnd,bmd->bnm', q, k) * self.scale_param  # (bs*num_heads, seqlen, seqlen)

        if mask is not None:
            pad_mask = mask.repeat_interleave(self.num_heads, dim=0)
            attention_scores = attention_scores.masked_fill(pad_mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attn_output = torch.einsum('bnm,bmd->bnd', attention_weights, v)  # (bs*num_heads, seqlen, latent_dim)

        attn_output = self._reshape_from_heads(attn_output)  # (bs, seqlen, num_heads * latent_dim)
        attn_output = self.fc_out(attn_output)  # (bs, seqlen, emb_size)

        if output_attentions:
            # Возвращаем weights в форме [bs*num_heads, seqlen_q, seqlen_kv], reshape если нужно
            return attn_output, attention_weights.view(bs, self.num_heads, seqlen, seqlen)
        return attn_output
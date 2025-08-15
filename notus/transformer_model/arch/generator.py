import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderLayerCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayerCrossAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, self_attn_mask=None, cross_attn_mask=None, non_masked_indices=None):
        # Если компрессия: используем non_masked_indices для gather/scatter
        if non_masked_indices is not None:
            orig_x = x  # Сохраняем оригинал для scatter
            x = torch.gather(x, 1, non_masked_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [batch, non_masked_len, d_model]

        # Self-attention (на compressed x)
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, attn_mask=self_attn_mask, need_weights=True)
        attn_out = self.dropout(self_attn_out)
        if non_masked_indices is not None:
            # Scatter обратно в оригинальные позиции
            attn_out = torch.zeros_like(orig_x).scatter_(1, non_masked_indices.unsqueeze(-1).expand(-1, -1, attn_out.size(-1)), attn_out)
        x = self.norm1(x + attn_out) if non_masked_indices is None else self.norm1(orig_x + attn_out)

        # Cross-attention (на full x, так как context короткий)
        cross_attn_out, cross_attn_weights = self.cross_attn(x, context, context, attn_mask=cross_attn_mask, need_weights=True)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x, self_attn_weights, cross_attn_weights

class DecoderCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super(DecoderCrossAttention, self).__init__()
        self.layers = nn.ModuleList([DecoderLayerCrossAttention(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.num_layers = n_layers

    def forward(self, x, context, self_attn_mask=None, cross_attn_mask=None, non_masked_indices=None):
        self_attn_weights = []
        cross_attn_weights = []

        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, context, self_attn_mask, cross_attn_mask, non_masked_indices)
            self_attn_weights.append(self_attn_w)
            cross_attn_weights.append(cross_attn_w)

        return x, self_attn_weights, cross_attn_weights

class FileRecoveryTransformer(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_layers=6, max_seq_len=4096, embedding_tensor=1024, vocab_size=267, d_ff=4096, dropout=0.1):
        super().__init__()
        self.file_projection = nn.Linear(embedding_tensor, d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, d_model))
        self.decoder = DecoderCrossAttention(d_model, nhead, d_ff, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, 267)  # 267 для байтов
        self.norm = nn.LayerNorm(embedding_tensor)

    def forward(self, emb, x, mask_non_masked=None, mask_x=None):
        # emb — корректировочный тензор (глобальный вектор)
        emb = self.norm(emb)
        file_vec = self.file_projection(emb)  # [batch, d_model]
        file_vec = file_vec.unsqueeze(1)  # [batch, 1, d_model] для cross-attention

        # Эмбеддинги последовательности
        seq_emb = self.token_emb(x)  # [batch, seq_len]
        seq_emb += self.pos_enc[:x.size(1), :].unsqueeze(0)  # [batch, seq_len, d_model]

        # Подготовка non_masked_indices (позиции, где mask_non_masked == 0, т.е. не маска)
        non_masked_indices = None
        if mask_non_masked is not None:
            non_masked_indices = (mask_non_masked == 0).nonzero(as_tuple=False)[:, 1].view(x.size(0), -1)  # [batch, non_masked_len]

        # Декодер с компрессией
        out, _, _ = self.decoder(seq_emb, file_vec, self_attn_mask=mask_x, non_masked_indices=non_masked_indices)

        return self.out_proj(out)  # [batch, seq_len, 267]
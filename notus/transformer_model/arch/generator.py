import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return context, attn

    def forward(self, query, key, value, mask=None):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        context, attn = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(context)
        return output, attn

class DecoderLayerCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayerCrossAttention,self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, self_attn_mask=None, cross_attn_mask=None):
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out, cross_attn_weights = self.cross_attn(x, context, context, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x, self_attn_weights, cross_attn_weights

class DecoderCrossAttention(nn.Module):
    def __init__(self, encoder_layer, n_enc_layer):
        super(DecoderCrossAttention, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(n_enc_layer)])
        self.num_layers = n_enc_layer

    def forward(self, x, context, self_attn_mask=None, cross_attn_mask=None):
        self_attn_weights = []
        cross_attn_weights = []

        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, context, self_attn_mask, cross_attn_mask)
            self_attn_weights.append(self_attn_w)
            cross_attn_weights.append(cross_attn_w)

        return x, self_attn_weights, cross_attn_weights

class FileTransformerBlock(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=6, max_seq_len=4096, embedding_tensor=1024, vocab_size=267):
        super().__init__()

        self.file_projection = nn.Linear(embedding_tensor, d_model)

        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, d_model))

        self.decoder = DecoderCrossAttention(
            DecoderLayerCrossAttention(
                d_model=d_model,
                n_heads=nhead,
                d_ff=d_model*4),
            num_encoder_layers
        )

        self.flatten = nn.Flatten()
        self.out_proj = nn.Linear(d_model, 256)
        self.norm = nn.LayerNorm(embedding_tensor)

    def forward(self, emb, x, mask_x=None):
        emb = self.norm(emb)
        file_vec = self.file_projection(emb.flatten(1))

        seq_emb = self.token_emb(x)
        seq_emb += self.pos_enc[:x.size(0), None, :]


        out, _, _ = self.decoder(seq_emb, file_vec, self_attn_mask=mask_x)
        return self.out_proj(out)

# TODO: create LightningModule train class
# class TrainGenerator(pl.LightningModule):
#     def __init__(self, encoder: FileTransformerBlock):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.loss_fn = nn.CrossEntropyLoss()
#
#     def training_step(self, batch, batch_idx):
#         x = batch['chunks']['labels']
#         z, mu, log = self.encoder(x)
#         logits = self.decoder(z)
#
#         recon_loss = self.loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
#         kl_loss = -0.5 * torch.sum(1 + log - mu.pow(2) - log.exp())
#         loss = recon_loss + kl_loss
#
#         self.log("train_loss", loss)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
#         return optimizer
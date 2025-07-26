import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
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

class FileTransformerBlock(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=6, max_seq_len=4096, embedding_tensor=1024, vocab_size=267):
        super().__init__()

        self.file_projection = nn.Linear(embedding_tensor, d_model)

        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, d_model))

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.flatten = nn.Flatten()
        self.out_proj = nn.Linear(d_model, 256)
        self.norm = nn.LayerNorm(embedding_tensor)

    def forward(self, emb, x):
        emb = self.norm(emb)
        file_vec = self.file_projection(emb.flatten(1))

        seq_emb = self.token_emb(x)
        seq_emb += self.pos_enc[:x.size(0), None, :]

        seq_emb += file_vec.unsqueeze(0)

        out = self.transformer(seq_emb)
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
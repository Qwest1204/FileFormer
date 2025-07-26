import torch
import torch.nn as nn
import pytorch_lightning as pl

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
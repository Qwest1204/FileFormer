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

class ValueTransformerBlock(FileTransformerBlock):
    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=6, max_seq_len=4096, embedding_tensor=1024,
                 vocab_size=267, correction_tensor=1024):
        super().__init__(d_model, nhead, num_encoder_layers, max_seq_len, embedding_tensor, vocab_size)


        self.l1 = nn.Linear(d_model, correction_tensor)
        self.l2 = nn.Linear(max_seq_len, 1)


    def forward(self, x):
        seq_emb = self.token_emb(x)
        seq_emb += self.pos_enc[:x.size(0), None, :]

        out = self.transformer(seq_emb)
        x = self.l1(out)
        x = x.permute(0,2,1)
        x = self.l2(x)
        x = x.permute(0, 2, 1)
        x = x.squeeze(dim=1)
        return x

class FileTransformer(pl.LightningModule):
    def __init__(self, modelc, modely, ignore_index):
        super().__init__()
        self.modelc = modelc
        self.modely = modely
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def training_step(self, batch, batch_idx):
        chunks = batch['chunks']

        original_files_tokens = batch['full_file_tokens'][0]

        outputs = torch.zeros((1, 512), dtype=torch.float32).to('mps')

        for FFT in original_files_tokens:
            outputs += self.modelc(FFT.unsqueeze(0))

        logits = self.modely(
            outputs,
            chunks['masked_ids']
        )
        loss = self.loss_fn(logits.view(-1, logits.size(-1)),
                            chunks['origin_ids'].view(-1),)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

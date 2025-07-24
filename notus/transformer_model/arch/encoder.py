import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class Encoder(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=6, max_seq_len=512,
                 vocab_size=267, latent_dim=512):
        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, d_model))

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Слои для VAE: получение mu и logvar
        self.mu_layer = nn.Linear(d_model, latent_dim)
        self.logvar_layer = nn.Linear(d_model, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_emb = self.token_emb(x)  # [batch_size, seq_len, d_model]
        seq_emb = seq_emb + self.pos_enc[:x.size(1), :]  # Добавление позиционного кодирования

        out = self.transformer(seq_emb)  # [batch_size, seq_len, d_model]

        # Средний пулинг по оси seq_len
        x = out.mean(dim=1)  # [batch_size, d_model]

        # Получение mu и logvar
        mu = self.mu_layer(x)  # [batch_size, latent_dim]
        logvar = self.logvar_layer(x)  # [batch_size, latent_dim]

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_decoder_layers=6, max_seq_len=512,
                 vocab_size=267, latent_dim=512):
        super().__init__()

        # Входной слой: из скрытого пространства в промежуточное
        self.input_proj = nn.Linear(latent_dim, d_model * max_seq_len)
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4,
                dropout=0.1
            ),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Выходной слой для проекции в пространство словаря
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов для стабилизации обучения."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Инициализация позиционного кодирования
        nn.init.normal_(self.pos_enc, mean=0.0, std=0.02)

    def forward(self, z, tgt=None):
        """
        z: [batch_size, latent_dim] - скрытое представление из энкодера
        tgt: [batch_size, seq_len] - целевая последовательность (для обучения, опционально)
        """
        # Преобразование скрытого пространства в последовательность
        x = self.input_proj(z)  # [batch_size, d_model * max_seq_len]
        x = x.view(-1, self.max_seq_len, self.d_model)  # [batch_size, max_seq_len, d_model]

        # Если tgt предоставлен (обучение с teacher-forcing), используем его
        if tgt is not None:
            x = self.token_emb(tgt) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]

        # Добавление позиционного кодирования
        x = x + self.pos_enc[:x.size(1), :]  # [batch_size, seq_len, d_model]

        # Декодирование
        output = self.transformer(x)  # [batch_size, seq_len, d_model]

        # Проекция в пространство словаря
        logits = self.output_proj(output)  # [batch_size, seq_len, vocab_size]

        return logits


class EncoderDecoder(pl.LightningModule):
    def __init__(self, encoder, decoder, beta=0.5, annealing_epochs=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.beta = beta
        self.annealing_epochs = annealing_epochs
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    def training_step(self, batch, batch_idx):
        x = batch['origin']
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        logits = self.decoder(z, tgt=x)

        recon_loss = self.loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        beta = min(1.0, self.current_epoch / self.annealing_epochs) * self.beta
        loss = recon_loss + beta * kl_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("kl_loss", kl_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['origin']
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        logits = self.decoder(z, tgt=x)

        recon_loss = self.loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl_loss

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

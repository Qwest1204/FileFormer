import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import numpy as np

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.channels = channels
        self.dtype_override = dtype_override

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.channels),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc

class Encoder(nn.Module):
    def __init__(self, d_model=1024, hidden_dim=256, nhead=8, num_encoder_layers=6, max_seq_len=512,
                 vocab_size=267, latent_dim=512):
        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_enc = PositionalEncoding1D(d_model)

        self.conv = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim*4,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Слои для VAE: получение mu и logvar
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_emb = self.token_emb(x)  # [batch_size, seq_len, d_model]
        seq_emb = self.pos_enc(seq_emb.unsqueeze(0))  # Добавление позиционного кодирования

        x = self.conv(seq_emb.permute(0, 2, 1)).permute(0, 2, 1)

        out = self.transformer(x)  # [batch_size, seq_len, d_model]

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
        self.pos_enc = PositionalEncoding1D(d_model)
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
        x = self.pos_enc(x)  # [batch_size, seq_len, d_model]

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

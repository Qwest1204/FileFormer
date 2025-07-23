import torch
import torch.nn as nn
import pytorch_lightning as pl

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
        z = self.reparameterize(mu, logvar)  # [batch_size, latent_dim]

        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, d_model=1024, max_seq_len=512, vocab_size=267, latent_dim=512):
        super().__init__()

        # Входной слой: из скрытого пространства в промежуточное
        self.input_proj = nn.Linear(256, latent_dim)

        # Слой для расширения до размерности последовательности
        self.expand_seq = nn.Linear(latent_dim, max_seq_len * d_model)

        # Нормализация для стабилизации
        self.norm = nn.LayerNorm(d_model)

        # Выходной слой: проекция в пространство словаря
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Позиционное кодирование
        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, d_model))

        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, z):
        # z: [batch_size, 256]
        # seq_len: текущая длина последовательности (для поддержки переменной длины)

        # Преобразование скрытого представления
        x = self.input_proj(z)  # [batch_size, hidden_dim]
        x = torch.relu(x)  # Нелинейность

        # Расширение до последовательности
        x = self.expand_seq(x)  # [batch_size, max_seq_len * d_model]
        x = x.view(-1, self.max_seq_len, self.d_model)  # [batch_size, max_seq_len, d_model]

        # Ограничение длины последовательности
        x = x[:, :self.max_seq_len, :]  # [batch_size, seq_len, d_model]

        # Добавление позиционного кодирования
        x = x + self.pos_enc[:self.max_seq_len, :]  # [batch_size, seq_len, d_model]

        # Нормализация
        x = self.norm(x)  # [batch_size, seq_len, d_model]

        # Проекция в пространство словаря
        logits = self.output_proj(x)  # [batch_size, seq_len, vocab_size]

        return logits


class EncoderDecoder(pl.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder,):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x = batch['chunks']['labels']
        z, mu, log = self.encoder(x)
        logits = self.decoder(z)

        recon_loss = self.loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
        kl_loss = -0.5 * torch.sum(1 + log - mu.pow(2) - log.exp())
        loss = recon_loss + kl_loss

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

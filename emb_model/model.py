import torch
import torch.nn as nn

class EmbModel(nn.Module):
    def __init__(self, vocab_size, token_embed_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        # Энкодер
        self.encoder_embed = nn.Embedding(vocab_size, token_embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(32 * token_embed_dim, 1024),
            #nn.BatchNorm1d(512),# Объединяем 16 токенов в один вектор
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            #nn.Dropout(0.2),
            #nn.Linear(128, 64)                     # Сжатие до 64
        )

        # Декодер
        self.decoder = nn.Sequential(
            #nn.Linear(64, 128),
            #nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(1024, 32 * vocab_size)         # Восстановление 16 токенов
        )

    def forward(self, x):
        # x: [batch_size, 16] (индексы токенов)
        embeddings = self.encoder_embed(x)          # [batch_size, 16, token_embed_dim]
        flattened = embeddings.view(x.size(0), -1)  # [batch_size, 16 * token_embed_dim]
        encoded = self.encoder(flattened)           # [batch_size, 64]
        decoded = self.decoder(encoded)             # [batch_size, 16 * vocab_size]
        return decoded.view(-1, 32, self.vocab_size)     # [batch_size, 16, vocab_size]
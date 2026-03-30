import torch.nn as nn
from fileformer.transformer_model.arch import attention
from fileformer.transformer_model.arch import pe

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class FileFormerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dim_ff: int,
                 dropout: float,
                 qkv_bias: bool,
                 ):
        super(FileFormerBlock, self).__init__()
        self.mha = attention.MultiHeadAttention(dim, num_heads, qkv_bias)
        self.ff = FeedForward(dim, dim_ff)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.mha(self.ln1(x), self.ln1(x), self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class FileFormer(nn.Module):
    def __init__(self, vocab_size, embed_size, n_heads, n_layers, drop_rate):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.positional_encoding = pe.RotaryPositionalEmbeddings(embed_size)

        # Create a list of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            # Each transformer block consists of multi-head attention and feed-forward layers
            FileFormerBlock(embed_size, n_heads, embed_size * 4, drop_rate, False)
            for _ in range(n_layers)  # Repeat for the number of layers specified in the config
        ])

        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(drop_rate)


    def forward(self, x, mask=None):
        # Step 1: Convert input token IDs to embeddings and add positional encodings
        x = self.dropout(self.positional_encoding(self.embedding(x)))

        # Step 2: Pass the embeddings through each transformer block
        for block in self.transformer_blocks:
            x = block(x, mask)  # Apply the transformer block with optional masking

        # Step 3: Project the final output to the vocabulary size
        return self.fc_out(x)  # Shape: (batch_size, seq_length, vocab_size)
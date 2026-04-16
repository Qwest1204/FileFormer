import torch.nn as nn
from .attention import MultiHeadAttention
from .pe import RotaryPositionalEmbeddings
from .mlp import MoELayer

class FileFormerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dim_ff: int,
                 num_experts: int,
                 top_k: int,
                 noisy_gating: bool,
                 dropout: float,
                 qkv_bias: bool,
                 is_causal: bool,
                 rank: int,
                 lora_alpha: float,
                 use_lora: bool

                 ):
        super(FileFormerBlock, self).__init__()
        self.mha = MultiHeadAttention(dim, num_heads, qkv_bias, enable_lora=use_lora, rank=rank, lora_alpha=lora_alpha)
        self.ff = MoELayer(d_model=dim, d_ff=dim_ff, num_experts=num_experts, top_k=top_k, noisy_gating=noisy_gating, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal

    def forward(self, x, mask=None):
        x = x + self.dropout(self.mha(query=self.ln1(x), key=self.ln1(x), value=self.ln1(x), mask=mask, is_causal=self.is_causal))
        ff_out, loss = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)
        return x, loss


class FileFormer(nn.Module):
    def __init__(self, vocab_size, embed_size, n_heads, n_layers, dropout, num_experts:int, top_k:int, noisy_gating:bool, rank: int, lora_alpha: float, use_lora: bool):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.positional_encoding = RotaryPositionalEmbeddings(embed_size)

        # Create a list of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            # Each transformer block consists of multi-head attention and feed-forward layers
            FileFormerBlock(dim=embed_size, num_heads=n_heads, dim_ff=embed_size * 4, dropout=dropout, qkv_bias=False, is_causal=False, use_lora=use_lora, rank=rank, lora_alpha=lora_alpha, num_experts=num_experts, top_k=top_k, noisy_gating=noisy_gating)
            for _ in range(n_layers)  # Repeat for the number of layers specified in the config
        ])

        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # Step 1: Convert input token IDs to embeddings and add positional encodings
        total_aux_loss = 0.0
        x = self.dropout(self.positional_encoding(self.embedding(x)))

        # Step 2: Pass the embeddings through each transformer block
        for block in self.transformer_blocks:
            x, loss = block(x, mask)  # Apply the transformer block with optional masking
            total_aux_loss += loss

        # Step 3: Project the final output to the vocabulary size
        return self.fc_out(x), total_aux_loss  # Shape: (batch_size, seq_length, vocab_size)
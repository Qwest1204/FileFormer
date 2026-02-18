import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, emb_size:int, dim_ff:int, activation_type:str, dropout:float):
        super(MLP, self).__init__()
        if activation_type == "relu":
            self.activation = nn.ReLU()
        if activation_type == "gelu":
            self.activation = nn.GELU()
        else:
            assert "Unknown activation type, avai: gelu, relu"
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            self.activation,
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.mlp(x)
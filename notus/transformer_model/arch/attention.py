import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim:int, head_dim:int):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim

        self.Q_layer = nn.Linear(embedding_dim, embedding_dim)
        self.K_layer = nn.Linear(embedding_dim, embedding_dim)
        self.V_layer = nn.Linear(embedding_dim, embedding_dim)
        self.out_fc = nn.Linear(embedding_dim, embedding_dim)
        self.scale_param = head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        bs, seqlen, dim = q.shape

        Q = self.Q_layer(q)     #[bs, seq_len, head_dim]
        K = self.K_layer(k)     #[bs, seq_len, head_dim]
        V = self.V_layer(v)     #[bs, seq_len, head_dim]

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale_param #[bs, seq_len, seq_len]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        return self.out_fc(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads,):
        super(MultiHeadAttention, self).__init__()
        assert emb_size % num_heads == 0, "Emb_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.Q_layer = nn.Linear(emb_size, emb_size)
        self.K_layer = nn.Linear(emb_size, emb_size)
        self.V_layer = nn.Linear(emb_size, emb_size)

        self.fc_out = nn.Linear(emb_size, emb_size)
        self.scale_param = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        bs, seqlen, dim = q.shape

        Q = self.Q_layer(q)
        K = self.K_layer(k)
        V = self.V_layer(v)

        #Reshape Q, K, V to (N, num_heads, seq_len, head_dim)
        Q = Q.view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale_param #[bs, seq_len, seq_len]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(bs, seqlen, dim)
        return self.fc_out(output)

class MultiQueryAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int):
        super(MultiQueryAttention, self).__init__()
        assert emb_size % num_heads == 0, "Emb_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.Q_layer = nn.Linear(emb_size, num_heads* self.head_dim)
        self.K_layer = nn.Linear(emb_size, self.head_dim)
        self.V_layer = nn.Linear(emb_size, self.head_dim)

        self.fc_out = nn.Linear(num_heads*self.head_dim, emb_size)
        self.scale_param = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        bs, seqlen, dim = q.shape
        _, seq_len_k, _ = k.shape

        Q = self.Q_layer(q) # (bs, seq_len, num_heads*head_dim)
        K = self.K_layer(k) # (bs, seq_len, head_dim)
        V = self.V_layer(v) # (bs, seq_len, head_dim)

        Q = Q.view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2) # (bs, seq_len, num_heads*head_dim) -> (bs, num_heads, seq_len, head_dim)
        K_t = K.unsqueeze(1).transpose(-1, -2) # (bs, seq_len, head_dim) -> (bs, 1, head_dim, seq_len_k)
        V_exp = V.unsqueeze(1)

        attention_scores = torch.matmul(Q, K_t) * self.scale_param #[bs, num_heads, seq_len_q, seq_len_k]
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1) # (bs, num_heads, seq_len_q, seq_len_k)
        output = torch.matmul(attention_weights, V_exp)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, self.num_heads * self.head_dim)
        return self.fc_out(output)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadLatentAttention, self).__init__()
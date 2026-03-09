import torch
import torch.nn.functional as F

def tensor_entropy(tokens):
    unique, counts = torch.unique(tokens, return_counts=True)

    probs = counts.float() / counts.sum()

    entropy = -torch.sum(probs * torch.log(probs))

    return entropy

def average_output_entropy(model, x, padding_mask=None):

    with torch.no_grad():
        logits = model(x, padding_mask.to(torch.bool))                     # (bs, seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1)      # (bs, seq_len, vocab_size)
        entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # (bs, seq_len)

        if padding_mask is not None:
            avg_entropy = (entropy_per_token * padding_mask).sum() / padding_mask.sum()
        else:
            avg_entropy = entropy_per_token.mean()

    return avg_entropy.item()

def evaluation(model, x, pads):
    print(f"Shenon entropy {tensor_entropy(x)}")
    print(f"Model entropy {average_output_entropy(model, x, pads)}")


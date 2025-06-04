import os.path

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from transformer_model import build_transformer

from ray.train.torch import TorchTrainer
from ray.train import SyncConfig
import tempfile
import ray.train.torch

def train_fn(dataset, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("all OK")
    transformer = build_transformer(vocab_size=5667,
                                d_model=512,
                                max_seq_len=256,
                                d_ff=1024,
                                dropout=0.1,
                                n_layers=6,
                                n_heads=8,
                                factor=2)
    print("all OK")
    criterion = nn.CrossEntropyLoss(ignore_index=5)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    transformer.to(device)
    transformer.train()

    print("all OK")
    for epoch in tqdm(range(40)):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = transformer(input_ids, attention_mask)

            # Расчет потерь (сравниваем выход с входом)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), input_ids.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = {"loss": loss.item(), "epoch": epoch}
        torch.save(transformer.state_dict(),
                       f"model{epoch}.pt"
                       )
        print(metrics)

dataset = torch.load("dataset.pt", weights_only=False)

train_fn(dataset, "cuda")
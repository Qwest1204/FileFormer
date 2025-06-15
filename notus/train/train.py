from notus import build_transformer
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm


def train_compress(epochs, trainloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = build_transformer(vocab_size=5772,
                                    d_model=1024,
                                    max_seq_len=2048,
                                    d_ff=4096,
                                    dropout=0.1,
                                    n_layers=10,
                                    n_heads=16,
                                    factor=64,
                                    compress=True)
    trainloader.to(device)
    transformer.train()
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss(ignore_index=5)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    for epoch in range(epochs):
        dataloader = tqdm(trainloader)
        loss_compute = []
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
            loss_compute.append(loss.item())
            dataloader.set_description("loss: %s" % str(sum(loss_compute)/len(loss_compute)))
        torch.save({"model_std": transformer.state_dict(),
                    "optimizer_std": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": loss.item()},
                   f"model{epoch}.pt"
                   )


def train_error_fix():
    pass

import torch
from tqdm import tqdm

def train_one_epoch(model, optimizer, loss_fn, train_dataloader, val_dataloader, device, epoch):
    model.train()
    train_loss = 0.0
    val_loss = 0.0

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")
    for tgt, src in pbar:
        tgt = tgt.to(device)
        src = src.to(device)

        optimizer.zero_grad()
        logits = model(src)

        logits = logits.reshape(-1, logits.size(-1))
        tgt = tgt.reshape(-1)  # [batch * (seq_len-1)]

        loss = loss_fn(logits, tgt)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

    avg_train_loss = train_loss / len(train_dataloader)

    log_msg = f"Epoch {epoch}: train_loss={avg_train_loss:.4f}"
    print(log_msg)

    avg_val_loss = 0

    if val_dataloader:

        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch} [Val]")

        model.eval()

        for tgt, src in val_pbar:
            tgt = tgt.to(device)
            src = src.to(device)

            logits = model(src)

            logits = logits.reshape(-1, logits.size(-1))
            tgt = tgt.reshape(-1)  # [batch * (seq_len-1)]

            loss = loss_fn(logits, tgt)

            val_loss += loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_dataloader)

        log_msg = f"Epoch {epoch}: avg_val_loss={avg_val_loss:.4f}"
        print(log_msg)

    return model, optimizer, avg_val_loss, avg_train_loss
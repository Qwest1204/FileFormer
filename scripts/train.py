
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fileformer import FileFormer
from fileformer.file_dataset import ENWIK8Dataset


def train_fileformer(
        model: FileFormer,
        train_dataset,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 3e-4,
        device: str = "cuda",
        save_dir: str = "checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)
    print(1)

    model = model.to(device)
    print(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print(1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=256)  # игнорируем pad_token_id
    print(1)

    for epoch in range(1, epochs + 1):
        # ===== Training =====
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for input_ids, padding_mask, _ in pbar:
            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)

            x = input_ids[:, :-1]
            y = input_ids[:, 1:]

            pad_mask = padding_mask[:, 1:]

            # Forward
            logits = model(x, pad_mask)  # [batch, seq_len-1, vocab_size]

            # Reshape для loss
            logits = logits.reshape(-1, logits.size(-1))  # [batch * (seq_len-1), vocab_size]
            y = y.reshape(-1)  # [batch * (seq_len-1)]

            loss = criterion(logits, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        # ===== Logging =====
        log_msg = f"Epoch {epoch}: train_loss={avg_train_loss:.4f}"
        print(log_msg)

        # ===== Save checkpoint =====
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
        }
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    return model


# ===== Пример использования =====
if __name__ == "__main__":
    VOCAB_SIZE = 257
    EMBED_SIZE = 256
    MAX_SEQ_LEN = 16828
    N_HEADS = 4
    N_LAYERS = 6
    DROP_RATE = 0.0
    print(1)
    # Датасет
    train_dataset = ENWIK8Dataset(
        file_path="/Users/daniilogorodnikov/PycharmProjects/Notus/enwik8",
        seq_len=MAX_SEQ_LEN,
        overlap=0,
        cache_dir="/Users/daniilogorodnikov/PycharmProjects/Notus/cache"
    )
    print(1)
    # Модель
    model = FileFormer(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        drop_rate=DROP_RATE
    )

    # Обучение
    train_fileformer(
        model=model,
        train_dataset=train_dataset,
        epochs=10,
        batch_size=2,
        lr=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="checkpoints"
    )
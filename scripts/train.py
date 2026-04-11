import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FileFormer, ByteLevelTokenizer
from dataset import FileDataset
from training import train_one_epoch

tokenizer = ByteLevelTokenizer()

VOCAB_SIZE = tokenizer.vocab_size
EMBED_SIZE = 256
SEQ_LEN = 8096
VAL_LEN = int(SEQ_LEN/2)
N_HEADS = 4
N_LAYERS = 6
DROP_RATE = 0.07
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-5
CLIP_GRAD_NORM = 10.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "../checkpoints/"

train_dataset = FileDataset(file_path="../data/raw/<file>", seq_len=SEQ_LEN, mask_prob=DROP_RATE, cache_dir="../data/.cache")
val_dataset = FileDataset(file_path="../data/raw/<file>", seq_len=VAL_LEN, mask_prob=DROP_RATE, cache_dir="../data/.cache")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = FileFormer(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        drop_rate=DROP_RATE
    ).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=1)  # игнорируем pad_token_id

def main(
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        val_dataloader,
        device,
        epochs,
        clip_grad_norm,
        save_dir,
    ):

    for epoch in range(1, epochs + 1):

        model, optimizer, avg_train_loss, avg_val_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            epoch=epoch,
            clip_grad_norm=clip_grad_norm,
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE,
                "embed_size": EMBED_SIZE,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "drop_rate": DROP_RATE
            }
        }
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    return 0


if __name__ == "__main__":
    main(model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        DEVICE,
        EPOCHS,
        CLIP_GRAD_NORM,
        SAVE_DIR
    )
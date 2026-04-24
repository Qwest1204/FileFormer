import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FileRWKV
from tokenizer import ByteLevelTokenizer
from dataset import MultiFileDataset
from trainig import train_one_epoch

os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_CTXLEN"] = "1024"
os.environ["RWKV_FLOAT_MODE"] = "bf16"

torch.backends.cuda.enable_flash_sdp(True)
tokenizer = ByteLevelTokenizer()

class Args:
    vocab_size = tokenizer.vocab_size
    ctx_len = 4000
    n_layer = 12
    n_embd = 512
    head_size_a = 64
    head_size_divisor = 8


args = Args()

BATCH_SIZE = 2
EPOCHS = 10
LR = 3e-4
CLIP_GRAD_NORM = 10.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints/"

train_dataset = MultiFileDataset('data/raw/', ['i'], 4000, 'data/chace/ch2.npy')
val_dataset = MultiFileDataset('data/raw/', ['j'], 2000, 'data/chace/ch1.npy')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FileRWKV(args).to(device).bfloat16()
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # игнорируем pad_token_id

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
            "config":   {  "vocab_size" : tokenizer.vocab_size,
                            "ctx_len" : 4000,
                            "n_layer" : 12,
                            "n_embd" : 512,
                            "head_size_a" : 64,
                            "head_size_divisor" : 8,
        }}
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
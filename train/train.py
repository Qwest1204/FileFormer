from notus import build_transformer, FileDataset, ByteLevelTokenizer, FileTransformer
import torch
import torch.optim as optim
import torch.nn as nn
from torch.functional import F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from torch.utils.tensorboard import SummaryWriter

# ======================== КОНФИГУРАЦИЯ ========================
EPOCHS = 20
BATCH_SIZE = 1
SEQ_LEN = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
SAVE_DIR = "./saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN_ID = 3
NUM_WORKERS = min(4, os.cpu_count())

os.makedirs(SAVE_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True


# ==============================================================

def init_models():
    transformer_correction = FileTransformer(
        d_model=2048,
        nhead=8,
        num_encoder_layers=4,
        max_seq_len=512,
        embedding_tensor=1024,
        vocab_size=267,

    ).to(DEVICE)

    correction_data = build_transformer(
        vocab_size=267,
        max_seq_len=SEQ_LEN,
        d_model=32,
        d_ff=2048,
        n_layers=8,
        n_heads=16,
        compress=True,
        window_size=32
    ).to(DEVICE)

    # Компиляция для PyTorch 2.0+
    if hasattr(torch, 'compile'):
        transformer_correction = torch.compile(transformer_correction)
        correction_data = torch.compile(correction_data)

    return transformer_correction, correction_data


def save_checkpoint(epoch, models, optimizer, loss, is_best=False):
    state = {
        "epoch": epoch,
        "loss": loss,
        "optimizer": optimizer.state_dict(),
    }
    for i, model in enumerate(models):
        state[f"model_{i}_state_dict"] = model.state_dict()

    filename = f"model_epoch_{epoch + 1}.pt"
    if is_best:
        filename = f"best_model.pt"

    torch.save(state, os.path.join(SAVE_DIR, filename))
    return filename


# ======================== ОСНОВНОЙ БЛОК ========================
if __name__ == "__main__":
    print("# Инициализация данных")
    tokenizer = ByteLevelTokenizer()
    dataset = FileDataset("/home/qwest/app/", SEQ_LEN, tokenizer)
    # torch.save(dataset, "glob_data.pt")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dataset.collate_fn  # Упрощенная коллация
    )

    print("# Инициализация моделей")
    transformer_correction, correction_data = init_models()

    print("# Оптимизация и планировщик")
    all_params = list(transformer_correction.parameters()) + list(correction_data.parameters())
    optimizer = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(SAVE_DIR, "logs", timestamp))

    print("# Начало обучения")
    best_loss = float('inf')
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_samples = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        transformer_correction.train()
        correction_data.train()

        for batch in progress_bar:
            try:
                # Обработка чанков
                chunks = batch['chunks']
                total_chunks = len(chunks['input_ids'])

                # Инициализация накопителей
                total_loss = 0.0

                # Обработка чанков пакетами по 8 для баланса скорости и памяти
                for i in range(0, total_chunks, 8):
                    # Выборка блока чанков
                    chunk_batch = {
                        'origin_ids': chunks['origin_ids'][i:i + 8].to(DEVICE, non_blocking=True),
                        'masked_ids': chunks['input_ids'][i:i + 8].to(DEVICE, non_blocking=True),
                        'attention_mask': chunks['attention_mask'][i:i + 8].to(DEVICE, non_blocking=True)
                    }

                    batch_size = chunk_batch['origin_ids'].size(0)

                    # Mixed precision forward pass
                    bias = correction_data(
                        chunk_batch['origin_ids'],
                        chunk_batch['attention_mask']
                    )
                    logits = transformer_correction(
                        bias,
                        chunk_batch['masked_ids']
                    )

                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        chunk_batch['origin_ids'].view(-1),
                        ignore_index=PAD_TOKEN_ID
                    )
                    loss = loss * batch_size

                    # Оптимизированный backward pass
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    grad_norm = nn.utils.clip_grad_norm_(all_params, 1.0)

                    optimizer.step()

                    total_loss += loss.item()

                # Обновление статистики
                epoch_loss += total_loss
                epoch_samples += total_chunks
                global_step += 1

                avg_loss = total_loss / total_chunks
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

                # Логирование
                writer.add_scalar('Loss/train', avg_loss, global_step)
                writer.add_scalar('Grad/norm', grad_norm, global_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    tqdm.write(f"OOM error: {str(e)}")
                    torch.cuda.empty_cache()
                continue

        # Финализация эпохи
        avg_epoch_loss = epoch_loss / epoch_samples
        scheduler.step(avg_epoch_loss)
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch + 1)
        print(f"\nEpoch {epoch + 1} | Loss: {avg_epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Сохранение чекпоинта
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(
                epoch,
                [transformer_correction, correction_data],
                optimizer,
                avg_epoch_loss,
                is_best=True
            )
            print(f"Saved BEST model with loss: {best_loss:.4f}")

        # Периодическое сохранение
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            save_checkpoint(
                epoch,
                [transformer_correction, correction_data],
                optimizer,
                avg_epoch_loss
            )

    writer.close()
    print("Обучение успешно завершено!")
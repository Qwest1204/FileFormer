from notus import build_transformer, UpscaleNet, Tokenizer, FileDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

# ========================
# Конфигурация
# ========================
EPOCHS = 50
BATCH_SIZE = 7
SEQ_LEN = 2048
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
SAVE_DIR = "./saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN_ID = 20  # Убедитесь, что ID соответствует pad token в токенизаторе

# Создать директорию для сохранения моделей
os.makedirs(SAVE_DIR, exist_ok=True)

# ========================
# Инициализация данных
# ========================
tokenizer = Tokenizer()
tokenizer.load_vocab_and_merges('vocab.json', 'merges.txt')
dataset = FileDataset("/Users/daniilogorodnikov/dataset/app/", SEQ_LEN, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# ========================
# Инициализация моделей
# ========================
model_config_base = {
    "vocab_size": 5772,
    "max_seq_len": SEQ_LEN,
    "dropout": 0.1
}

transformer_correction = build_transformer(
    **model_config_base,
    d_model=2048,
    d_ff=2048,
    n_layers=10,
    n_heads=16,
    factor=1,
    compress=False,
).to(DEVICE)

correction_data = build_transformer(
    **model_config_base,
    d_model=32,
    d_ff=32,
    n_layers=4,
    n_heads=4,
    factor=128,
    compress=True,
).to(DEVICE)

up_net = UpscaleNet().to(DEVICE)

# ========================
# Оптимизация и планировщик
# ========================
all_params = list(transformer_correction.parameters()) + \
             list(correction_data.parameters()) + \
             list(up_net.parameters())

optimizer = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3,
)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

# ========================
# Обучение
# ========================
best_loss = float('inf')
start_epoch = 0
timestamp = time.strftime("%Y%m%d_%H%M%S")

for epoch in range(start_epoch, EPOCHS):
    epoch_loss = 0.0
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        leave=False
    )

    transformer_correction.train()
    correction_data.train()
    up_net.train()

    for batch in progress_bar:
        # Перемещение данных на устройство
        origin_ids = batch['origin_ids'].to(DEVICE, non_blocking=True)
        masked_ids = batch['masked_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask_masked_ids'].to(DEVICE, non_blocking=True)

        # Forward pass
        encoded_correction = transformer_correction.encode(masked_ids, attention_mask)
        bias = correction_data.encode(origin_ids, attention_mask)
        up_bias = up_net(bias)
        corrected = encoded_correction + up_bias
        decoded = transformer_correction.decode(corrected, attention_mask)
        logits = transformer_correction.project(decoded)

        # Расчет потерь
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            origin_ids.view(-1)
        )

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)  # Gradient clipping
        optimizer.step()

        # Обновление статистик
        epoch_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    # Вычисление средней потери за эпоху
    avg_epoch_loss = epoch_loss / len(dataloader)
    scheduler.step(avg_epoch_loss)
    print(f"\nEpoch {epoch + 1} | Avg Loss: {avg_epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Сохранение лучшей модели
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        checkpoint = {
            "transformer_correction": transformer_correction.state_dict(),
            "correction_data": correction_data.state_dict(),
            "up_net": up_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_epoch_loss,
            "config": model_config_base
        }
        torch.save(
            checkpoint,
            os.path.join(SAVE_DIR, f"best_model_{timestamp}.pt")
        )
        print(f"Saved best model with loss: {best_loss:.4f}")

    # Периодическое сохранение
    if (epoch + 1) % 5 == 0:
        torch.save(
            checkpoint,
            os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}_{timestamp}.pt")
        )

print("Training completed!")
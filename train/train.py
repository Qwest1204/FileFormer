from notus import build_transformer, UpscaleNet, Tokenizer, FileDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Для синхронной обработки ошибок
os.environ['TORCH_USE_CUDA_DSA'] = "1"

print("# ========================")
print("# Конфигурация")
print("# ========================")
EPOCHS = 20
BATCH_SIZE = 1
SEQ_LEN = 2048
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
SAVE_DIR = "./saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN_ID = 20  # Убедитесь, что ID соответствует pad token в токенизаторе

# Создать директорию для сохранения моделей
os.makedirs(SAVE_DIR, exist_ok=True)

print("# ========================")
print("# Инициализация данных")
print("# ========================")
tokenizer = Tokenizer()
tokenizer.load_vocab_and_merges('vocab.json', 'merges.txt')
dataset = FileDataset("/home/qwest/app/", SEQ_LEN, tokenizer)
torch.save(dataset, "dataset1.pt")
#dataset = torch.load("dataset.pt", weights_only=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

print("# ========================")
print("# Инициализация моделей")
print("# ========================")
model_config_base = {
    "vocab_size": 5772,
    "max_seq_len": SEQ_LEN,
    "dropout": 0.1
}

transformer_correction = build_transformer(
    **model_config_base,
    d_model=1024,
    d_ff=1024,
    n_layers=4,
    n_heads=4,
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

print("# ========================")
print("Оптимизация и планировщик")
print("# ========================")
all_params = list(transformer_correction.parameters()) + \
             list(correction_data.parameters()) + \
             list(up_net.parameters())

optimizer = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

def validate_tensor(tensor, name, vocab_size=None):
    """Проверка тензора на наличие проблем"""
    #print(f"[DEBUG] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

    # Проверка на NaN/inf
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN values detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf values detected in {name}")

    # Проверка индексов
    if vocab_size is not None and tensor.dtype == torch.long:
        unique_vals = torch.unique(tensor)
        print(f"[DEBUG] Unique values in {name}: min={unique_vals.min()}, max={unique_vals.max()}")

        if unique_vals.min() < 0:
            raise ValueError(f"Negative values in {name}")
        if unique_vals.max() >= vocab_size:
            raise ValueError(f"Value > vocab_size ({vocab_size}) in {name}: max={unique_vals.max()}")

# ========================
# Обучение
# ========================
best_loss = float('inf')
start_epoch = 0
timestamp = time.strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(os.path.join(SAVE_DIR, "logs", timestamp))  # Добавлено

step_counter = 0  # Добавлено: счетчик шагов для TensorBoard
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
        try:
            # Перемещение данных на устройство
            origin_ids = batch['origin_ids'].to(DEVICE, non_blocking=True)
            masked_ids = batch['masked_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask_masked_ids'].to(DEVICE, non_blocking=True)

            # =====================================
            # ДОБАВЛЕННЫЕ ПРОВЕРКИ ДАННЫХ
            # =====================================
            validate_tensor(origin_ids, "origin_ids", vocab_size=5772)
            validate_tensor(masked_ids, "masked_ids", vocab_size=5772)
            validate_tensor(attention_mask, "attention_mask")

            # Forward pass
            encoded_correction = transformer_correction.encode(masked_ids, attention_mask)
            validate_tensor(encoded_correction, "encoded_correction")

            # ИСПРАВЛЕНИЕ: использование правильной маски
            bias = correction_data.encode(origin_ids, attention_mask)  # Было attention_mask_masked_ids
            validate_tensor(bias, "bias")

            up_bias = up_net(bias)
            validate_tensor(up_bias, "up_bias")

            corrected = encoded_correction + up_bias
            validate_tensor(corrected, "corrected")

            decoded = transformer_correction.decode(corrected, attention_mask)
            validate_tensor(decoded, "decoded")

            logits = transformer_correction.project(decoded)
            validate_tensor(logits, "logits")

            # =====================================
            # ПРОВЕРКА ЛОССА ПЕРЕД ВЫЧИСЛЕНИЕМ
            # =====================================
            # Проверка совместимости размерностей
            if logits.shape[0] != origin_ids.shape[0] or logits.shape[1] != origin_ids.shape[1]:
                raise ValueError(
                    f"Shape mismatch: logits {logits.shape} vs origin_ids {origin_ids.shape}"
                )

            # Расчет потерь с обработкой ошибок
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                origin_ids.view(-1)
            )

            # Проверка лосса
            if torch.isnan(loss):
                raise ValueError("NaN loss detected")
            if torch.isinf(loss):
                raise ValueError("Inf loss detected")

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # =====================================
            # ГРАДИЕНТНЫЙ КЛИППИНГ С ПРОВЕРКОЙ
            # =====================================
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                raise ValueError(f"Invalid gradient norm: {grad_norm}")

            optimizer.step()

            # Обновление статистик
            epoch_loss += loss.item()
            step_counter += 1  # Добавлено: инкремент счетчика шагов

            # Логирование в TensorBoard каждые 20 шагов
            if step_counter % 20 == 0:  # Добавлено
                writer.add_scalar('Loss/train', loss.item(), step_counter)
                writer.add_scalar('Grad/norm', grad_norm, step_counter)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], step_counter)

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "grad_norm": f"{grad_norm:.4f}"
            })

        except Exception as e:
            print(f"\n[ERROR] Epoch {epoch} batch {id}: {str(e)}")
            print("Skipping batch...")
            # Дополнительная диагностика при ошибке
            print("Batch content:")
            for k, v in batch.items():
                print(f"{k}: shape={v.shape} dtype={v.dtype}")

            # Пропуск проблемного батча
            continue

    # Вычисление средней потери за эпоху
    avg_epoch_loss = epoch_loss / len(dataloader)
    writer.add_scalar('Loss/epoch_avg', avg_epoch_loss, epoch)  # Добавлено
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
            os.path.join(SAVE_DIR, f"best_model_localattention_{timestamp}.pt")
        )
        print(f"Saved best model with loss: {best_loss:.4f}")

    # Периодическое сохранение
    if (epoch + 1) % 5 == 0:
        torch.save(
            checkpoint,
            os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}_{timestamp}.pt")
        )

writer.close()  # Добавлено
print("Training completed!")
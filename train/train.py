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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Для синхронной обработки ошибок
os.environ['TORCH_USE_CUDA_DSA'] = "1"

print("# ========================")
print("# Конфигурация")
EPOCHS = 20
BATCH_SIZE = 1
SEQ_LEN = 4096
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
SAVE_DIR = "./saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN_ID = 3  # Убедитесь, что ID соответствует pad token в токенизаторе

# Создать директорию для сохранения моделей
os.makedirs(SAVE_DIR, exist_ok=True)

print("# Инициализация данных")
tokenizer = ByteLevelTokenizer()
dataset = FileDataset("/home/qwest/app/", SEQ_LEN, tokenizer)
torch.save(dataset, "dataset1.pt")
dataset = torch.load("dataset1.pt", weights_only=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

print("# Инициализация моделей")
model_config_base = {
    "vocab_size": 267,
    "max_seq_len": SEQ_LEN,
    "dropout": 0.1
}

transformer_correction = FileTransformer().to(DEVICE)

correction_data = build_transformer(
    vocab_size = 267,
    max_seq_len =  4096,
    dropout = 0.1,
    d_model=32,
    d_ff=2048,
    n_layers=8,
    n_heads=16,
    compress_factor=128,
    compress=True,
    window_size=32,
    return_attention=False
).to(DEVICE)

print("Оптимизация и планировщик")
all_params = list(transformer_correction.parameters()) + \
             list(correction_data.parameters())

optimizer = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3
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

print('# Обучение')
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

    for batch in progress_bar:
        try:
            # Перемещение данных на устройство
            origin_ids = batch['origin_ids'].to(DEVICE, non_blocking=True)
            masked_ids = batch['masked_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask_masked_ids'].to(DEVICE, non_blocking=True)

            # =====================================
            # ДОБАВЛЕННЫЕ ПРОВЕРКИ ДАННЫХ
            # =====================================
            validate_tensor(origin_ids, "origin_ids", vocab_size=267)
            validate_tensor(masked_ids, "masked_ids", vocab_size=267)
            validate_tensor(attention_mask, "attention_mask")

            # Forward pass
            bias = correction_data.forward(origin_ids.unsqueeze(0), attention_mask)
            validate_tensor(bias, "encoded_correction")

            data = transformer_correction.forward(bias, masked_ids.unsqueeze(1))
            validate_tensor(data, "encoded_correction")

            loss = F.cross_entropy(
                data.permute(1, 2, 0),
                origin_ids.unsqueeze(0),
                ignore_index=2# Целевые байты (0-255)
            )

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
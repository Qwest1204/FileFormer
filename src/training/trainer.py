import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    device: torch.device,
    epoch: int,
    clip_grad_norm: float = 1.0,
) -> Tuple[nn.Module, optim.Optimizer, float, float]:
    """
    Выполняет одну эпоху обучения (и опционально валидации) с подсчётом потерь
    только по замаскированным токенам.

    Args:
        model (nn.Module): Модель для обучения.
        optimizer (optim.Optimizer): Оптимизатор.
        loss_fn (nn.Module): Функция потерь (обычно CrossEntropyLoss).
        train_dataloader (DataLoader): DataLoader, возвращающий пары (target, source).
        val_dataloader (Optional[DataLoader]): DataLoader для валидации.
        device (torch.device): Устройство (CPU/GPU).
        epoch (int): Номер текущей эпохи.
        clip_grad_norm (float, optional): Максимальная норма градиента.

    Returns:
        Кортеж: (model, optimizer, avg_train_loss, avg_val_loss).
    """
    model.train()
    train_loss_accumulator = 0.0
    train_pp = 0.0

    # --------------------- Цикл обучения ---------------------
    train_progress = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch} [Train]",
        unit="batch",
        leave=False,
    )
    for target, source in train_progress:
        target = torch.tensor(target, dtype=torch.long).to(device, non_blocking=True)
        source = torch.tensor(source, dtype=torch.long).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Прямой проход
        logits, aux_loss = model(source)

        # Просто вычисляем потери – CrossEntropyLoss сам проигнорирует ignore_index
        loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
        loss = loss + 0.001 * aux_loss
        # Обратный проход и шаг оптимизатора
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        # Накопление потерь для статистики
        train_loss_accumulator += loss.item()
        pp = torch.exp(loss)
        train_pp += pp.item()
        train_progress.set_postfix({"loss": f"{loss.item():.4f}", "PP": f"{pp.item():.4f}"})

    avg_train_loss = train_loss_accumulator / len(train_dataloader)
    avg_pp = train_pp/len(train_dataloader)
    print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | PP: {avg_pp:.4f} ")

    # --------------------- Цикл валидации ---------------------
    avg_val_loss = 0.0
    if val_dataloader is not None:
        model.eval()
        val_loss_accumulator = 0.0
        val_pp = 0.0
        val_progress = tqdm(
            val_dataloader,
            desc=f"Epoch {epoch} [Val]  ",
            unit="batch",
            leave=False,
        )
        with torch.no_grad():
            for target, source in val_progress:
                target = target.to(device, non_blocking=True)
                source = source.to(device, non_blocking=True)
                logits, aux = model(source)
                loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1)) + 0.001 * aux
                val_loss_accumulator += loss.item()
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
                val_pp += torch.exp(loss).item()
        avg_val_loss = val_loss_accumulator / len(val_dataloader)
        avg_pp = val_pp/len(val_dataloader)
        print(f"Epoch {epoch:3d} | Val Loss:   {avg_val_loss:.4f} | PP:  {avg_pp:.4f}")

    return model, optimizer, avg_train_loss, avg_val_loss
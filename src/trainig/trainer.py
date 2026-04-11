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
    Perform one complete pass over the training dataset (and optionally the validation dataset).

    Args:
        model (nn.Module): The neural network model to train.
        optimizer (optim.Optimizer): Optimizer instance for updating model parameters.
        loss_fn (nn.Module): Loss function used for both training and validation.
        train_dataloader (DataLoader): DataLoader yielding (target, source) pairs for training.
        val_dataloader (Optional[DataLoader]): DataLoader for validation. If None, validation is skipped.
        device (torch.device): Device on which tensors should be placed (CPU or GPU).
        epoch (int): Current epoch number (used for logging).
        clip_grad_norm (float, optional): Maximum norm for gradient clipping. Defaults to 1.0.

    Returns:
        Tuple containing:
            - model: The trained model (updated in-place).
            - optimizer: The optimizer (state may have been updated).
            - avg_train_loss (float): Average training loss over the epoch.
            - avg_val_loss (float): Average validation loss (0.0 if no validation).
    """
    model.train()
    train_loss_accumulator = 0.0

    # --------------------- Training Loop ---------------------
    train_progress = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch} [Train]",
        unit="batch",
        leave=False,
    )
    for target, source in train_progress:
        target = target.to(device, non_blocking=True)
        source = source.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(source)

        # Reshape for loss computation: (batch * seq_len, vocab_size) vs (batch * seq_len)
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = target.reshape(-1)

        loss = loss_fn(logits_flat, target_flat)

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        # Accumulate and display loss
        train_loss_accumulator += loss.item()
        train_progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = train_loss_accumulator / len(train_dataloader)
    print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f}")

    # --------------------- Validation Loop ---------------------
    avg_val_loss = 0.0
    if val_dataloader is not None:
        model.eval()
        val_loss_accumulator = 0.0

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

                logits = model(source)

                logits_flat = logits.reshape(-1, logits.size(-1))
                target_flat = target.reshape(-1)

                loss = loss_fn(logits_flat, target_flat)

                val_loss_accumulator += loss.item()
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss_accumulator / len(val_dataloader)
        print(f"Epoch {epoch:3d} | Val Loss:   {avg_val_loss:.4f}")

    return model, optimizer, avg_train_loss, avg_val_loss
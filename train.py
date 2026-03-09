import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

from fileformer import utils, ENWIK8Dataset, ByteLevelTokenizer, Decoder, eval


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    logger.info(f"loading from {config_path}")
    return utils.load_config(config_path)


def setup_tokenizer() -> ByteLevelTokenizer:
    tokenizer = ByteLevelTokenizer()
    logger.info("ByteLevelTokenizer init")
    logger.info(f"  vocab size: {tokenizer.vocab_size}")
    pad_id = tokenizer.encode("<pad>")[0]
    mask_id = tokenizer.encode("<mask>")[0]
    logger.info(f"  ID <pad>: {pad_id}")
    logger.info(f"  ID <mask>: {mask_id}")
    return tokenizer


def setup_data(config: dict) -> tuple[ENWIK8Dataset, DataLoader]:

    logger.info("loading dataset ENWIK8")
    dataset = ENWIK8Dataset(**config['dataset'])
    logger.info(f"  size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=config['train']['shuffle'],
        num_workers=config['train']['num_workers'],
        pin_memory=config['train']['pin_memory']
    )
    logger.info(f"  dataloader loaded: {len(dataloader)}")
    return dataset, dataloader


def create_model(config: dict, device: torch.device) -> tuple[Decoder, torch.optim.Optimizer, torch.nn.Module]:

    logger.info("init Decoder")
    model = Decoder(**config['decoder']).to(device)

    # Применяем torch.compile, если доступно (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        logger.info("  model compiled with torch.compile")
    except Exception as e:
        logger.warning(f"  torch.compile failed, using uncompiled model: {e}")

    summary(model, depth=4)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr']
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

    logger.info(f"  on device: {device}")
    logger.info(f"  optim: AdamW, lr={config['train']['lr']}")
    logger.info(f"  loss: CrossEntropyLoss (ignore_index={1})")

    return model, optimizer, loss_fn


def train_epoch(
    model: Decoder,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    accumulation_steps: int
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"epoch {epoch+1}", leave=False)

    optimizer.zero_grad()
    running_loss = 0.0

    for i, batch in enumerate(progress_bar):
        x, padding, _ = batch

        input_ids = x[:, :-1].to(device)
        target_ids = x[:, 1:].to(device)
        attention_mask = torch.tensor(padding[:, :-1], dtype=torch.bool).to(device)

        logits = model(input_ids, attention_mask)          # (batch, seq_len, vocab_size)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.reshape(-1))

        # Нормируем loss на количество шагов накопления
        loss = loss / accumulation_steps
        loss.backward()

        running_loss += loss.item() * accumulation_steps  # восстанавливаем исходное значение loss для статистики

        # Шаг оптимизатора каждые accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            total_loss += running_loss
            avg_loss = total_loss / (i + 1)

            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            running_loss = 0.0

        # Вызов evaluation после каждого батча (как в исходном коде)
        eval.evaluation(model, x.to(device), padding.to(device))

    # Если остались ненулевые градиенты (когда число батчей не кратно accumulation_steps)
    if running_loss != 0.0:
        optimizer.step()
        optimizer.zero_grad()
        total_loss += running_loss

    return total_loss / num_batches


def train(config: dict) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")

    tokenizer = setup_tokenizer()
    _, dataloader = setup_data(config)
    model, optimizer, loss_fn = create_model(config, device)

    num_epochs = config['train']['num_epoch']
    save_dir = Path(config['train']['savedir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    accumulation_steps = config['train']['accumulation_steps']
    logger.info(f"gradient accumulation steps: {accumulation_steps}")

    logger.info(f"start {num_epochs} epochs")

    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, accumulation_steps)

        logger.info(f"ep {epoch+1} complete, avg loss: {epoch_loss:.4f}")

        checkpoint_path = save_dir / f"model-1M_epoch{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"  save in {checkpoint_path}")

    logger.info("complete! ")


def main():
    config = load_config('configs/config.yml')
    train(config)


if __name__ == "__main__":
    main()
import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
import torch.profiler
from fileformer import utils, ENWIK8Dataset, ByteLevelTokenizer, Decoder, eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    logger.info(f"loading from {config_path}")
    return utils.load_config(config_path)

def setup_tokenizer() -> ByteLevelTokenizer:
    tokenizer = ByteLevelTokenizer()
    logger.info("ByteLevelTokenizer init")
    logger.info(f" vocab size: {tokenizer.vocab_size}")
    pad_id = tokenizer.encode("<pad>")[0]
    mask_id = tokenizer.encode("<mask>")[0]
    logger.info(f" ID <pad>: {pad_id}")
    logger.info(f" ID <mask>: {mask_id}")
    return tokenizer

def setup_data(config: dict) -> tuple[ENWIK8Dataset, DataLoader]:
    logger.info("loading dataset ENWIK8")
    dataset = ENWIK8Dataset(**config['dataset'])
    logger.info(f" size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=config['train']['shuffle'], num_workers=config['train']['num_workers'], pin_memory=config['train']['pin_memory'])
    logger.info(f" dataloader loaded: {len(dataloader)}")
    return dataset, dataloader

def create_model(config: dict, device: torch.device) -> tuple[Decoder, torch.optim.Optimizer, torch.nn.Module]:
    logger.info("init Decoder")
    model = Decoder(**config['decoder']).to(device)
    summary(model, depth=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'])
    pad_token_id = 1
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    logger.info(f" on device: {device}")
    logger.info(f" optim: AdamW, lr={config['train']['lr']}")
    logger.info(f" loss: CrossEntropyLoss (ignore_index={pad_token_id})")
    return model, optimizer, loss_fn

def train_epoch(model: Decoder, dataloader: DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"epoch {epoch+1} (100 batches)", leave=False)
    for i, batch in enumerate(progress_bar):
        if i >= 100:
            break
        x, padding, _ = batch
        input_ids = x[:, :-1].to(device)
        target_ids = x[:, 1:].to(device)
        attention_mask = torch.tensor(padding[:, :-1], dtype=torch.bool).to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
        eval.evaluation(x.to(device), model, padding.to(device))
    return total_loss / min(100, len(dataloader))

def train(config: dict) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")
    tokenizer = setup_tokenizer()
    _, dataloader = setup_data(config)
    model, optimizer, loss_fn = create_model(config, device)
    logger.info("start profiler on 100 batches")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof:
        epoch_loss = train_epoch(model, dataloader, loss_fn, optimizer, device, 0)
    logger.info(f"100 batches done, avg loss: {epoch_loss:.4f}")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    prof.export_chrome_trace("profiler_trace.json")
    logger.info("profiler trace saved to profiler_trace.json")

def main():
    config = load_config('configs/config.yml')
    train(config)

if __name__ == "__main__":
    main()
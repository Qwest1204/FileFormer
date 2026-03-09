import torch
from torch.utils.data import DataLoader
from fileformer import ENWIK8Dataset, Decoder, utils

# Конфигурация (можно задать прямо в коде, но оставим загрузку для гибкости)
config = utils.load_config('configs/config.yml')

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Данные
dataset = ENWIK8Dataset(**config['dataset'])
dataloader = DataLoader(
    dataset,
    batch_size=config['train']['batch_size'],
    shuffle=False,                      # для профилирования порядок не важен
    num_workers=config['train']['num_workers'],
    pin_memory=config['train']['pin_memory']
)
print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")

# Модель, оптимизатор, loss
model = Decoder(**config['decoder']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'])
pad_token_id = 1
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
print("Model created")

# Профилирование на 100 батчах
model.train()
batch_iter = iter(dataloader)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=32),  # ~100 шагов
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(100):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        x, padding, _ = batch
        input_ids = x[:, :-1].to(device)
        target_ids = x[:, 1:].to(device)
        attention_mask = torch.tensor(padding[:, :-1], dtype=torch.bool).to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.reshape(-1))
        loss.backward()
        optimizer.step()

        prof.step()
        if step % 10 == 0:
            print(f"Step {step}, loss: {loss.item():.4f}")

print("Profiling completed. Trace saved in ./profiler_logs")
from notus import utils, FileFormer, FileDataset, FileFormerQuant
from torch.utils.data import DataLoader, random_split
import torch
from lightning import Trainer

configs = utils.load_config("config/config.yml")
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

device = configs['train']['device']

fileformer_fp32 = FileFormer.load_from_checkpoint(
    "model-epoch=02-step=110000.ckpt",
    loss_fn=loss_fn,
    config=configs,
    device=device
)

qat_model = FileFormerQuant(loss_fn, configs)
qat_model.encoder.load_state_dict(fileformer_fp32.encoder.state_dict())
qat_model.decoder.load_state_dict(fileformer_fp32.decoder.state_dict())


dataset = FileDataset(**configs['dataset'])

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_subset, val_subset = random_split(dataset, [train_size, val_size])

dataloader = DataLoader(val_subset, batch_size=4,
                        shuffle=True, num_workers=configs['train']['num_workers']
                        )

trainer = Trainer(
    accelerator='cpu',
    max_epochs=4,           # 5–15 эпох
    precision='32-true',     # QAT в fp32 с fake-quant
    log_every_n_steps=10,
)

trainer.fit(qat_model, dataloader)
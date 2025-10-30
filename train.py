from notus import utils, FileDataset, ByteLevelTokenizer, FileFormer
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

#tokenizer
tokenizer = ByteLevelTokenizer()

#configs
configs = utils.load_config("/Users/daniilogorodnikov/PycharmProjects/Notus/config/config.yml")

#dataset
dataset = FileDataset(**configs['dataset'])

dataloader = DataLoader(dataset, batch_size=configs['train']['batch_size'],
                        shuffle=True, num_workers=configs['train']['num_workers']
                        )

#loss
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",  # Папка для сохранения чекпоинтов
    filename="model-{epoch:02d}-{step:05d}",  # Шаблон имени файла
    every_n_train_steps=configs['train']['interval4save'],  # Сохранять каждые 1000 шагов
    save_top_k=5,  # Сохранять все чекпоинты (не ограничивать количество)
    save_weights_only=False,  # Сохранять полное состояние (модель, оптимизатор и т.д.)
    verbose=True  # Выводить информацию о сохранении
)

fileformer = FileFormer(loss_fn, configs)
trainer = L.Trainer(max_epochs=10, precision=configs['train']['precision'], callbacks=[checkpoint_callback])
trainer.fit(model=fileformer, train_dataloaders=dataloader)


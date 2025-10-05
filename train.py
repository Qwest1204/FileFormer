from notus import Encoder, Decoder, utils, FileDataset, Muon, ByteLevelTokenizer, eval
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

#tokenizer
tokenizer = ByteLevelTokenizer()

#configs
configs = utils.load_config("/Users/daniilogorodnikov/PycharmProjects/Notus/config/config.yml")

#dataset
dataset = FileDataset(**configs['dataset'])

dataloader = DataLoader(dataset, batch_size=configs['train']['batch_size'],
                        shuffle=True, num_workers=configs['train']['num_workers']
                        )

#models
encoder = Encoder(**configs['encoder'])
decoder = Decoder(**configs['decoder'])

#loss
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",  # Папка для сохранения чекпоинтов
    filename="model-{epoch:02d}-{step:05d}",  # Шаблон имени файла
    every_n_train_steps=configs['train']['interval4save'],  # Сохранять каждые 1000 шагов
    save_top_k=-1,  # Сохранять все чекпоинты (не ограничивать количество)
    save_weights_only=False,  # Сохранять полное состояние (модель, оптимизатор и т.д.)
    verbose=True  # Выводить информацию о сохранении
)

class FileFormer(L.LightningModule):
    def __init__(self, encoder, decoder, loss_fn, config):
        #comment
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.config = config
        self.automatic_optimization = True  # Для ручного управления оптимизацией

    def training_step(self, batch, batch_idx):
        tokens, masked_tokens, pads, hash, extention_tokenize = batch
        encoder_out = self.encoder(hash, extention_tokenize)

        decoder_out = self.decoder(masked_tokens, encoder_out, pads)

        loss = self.loss_fn(
            decoder_out.view(-1, self.config['encoder']['vocab_size']),
            tokens.view(-1)
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        opt = self.optimizers()
        opt.zero_grad()

        opt.step()

        if batch_idx % configs['train']['interval4save'] == 0:
            eval.evaluate(self.encoder, self.decoder, tokenizer, batch)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.config['train']['lr'],
        )

        return {
            "optimizer": optimizer,
        }

fileformer = FileFormer(encoder, decoder, loss_fn, configs)
trainer = L.Trainer(max_epochs=10, precision=configs['train']['precision'], callbacks=[checkpoint_callback])
trainer.fit(model=fileformer, train_dataloaders=dataloader)


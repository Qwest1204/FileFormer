from notus import Encoder, Decoder, utils, FileDataset, Muon
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import lightning as L

#configs
configs = utils.load_config("/Users/daniilogorodnikov/PycharmProjects/Notus/config/config.yml")

#dataset
dataset = FileDataset(**configs['dataset'])
dataset.prepare()
dataset_len = len(dataset)
dataloader = DataLoader(dataset, batch_size=configs['train']['batch_size'],
                        shuffle=True, num_workers=configs['train']['num_workers']
                        )

#models
encoder = Encoder(**configs['encoder'])
decoder = Decoder(**configs['decoder'])

#loss
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

class FileFormer(L.LightningModule):
    def __init__(self, encoder, decoder, loss_fn, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.config = config
        self.automatic_optimization = False  # Для ручного управления оптимизацией

    def training_step(self, batch, batch_idx):
        tokens, masked_tokens, pads, hash, extention_tokenize = batch

        # Проверка входных данных
        self._check_for_nan(tokens, "tokens")
        self._check_for_nan(masked_tokens, "masked_tokens")
        self._check_for_nan(hash, "hash")
        self._check_for_nan(extention_tokenize, "extention_tokenize")

        encoder_out = self.encoder(hash, extention_tokenize)
        self._check_for_nan(encoder_out, "encoder_out")

        decoder_out = self.decoder(masked_tokens, encoder_out, pads)
        self._check_for_nan(decoder_out, "decoder_out")

        # Расчет лосса
        loss = self.loss_fn(
            decoder_out.view(-1, self.config['encoder']['vocab_size']),
            tokens.view(-1)
        )

        # Проверка лосса и градиентов
        if torch.isnan(loss).any():
            self._log_nan_metrics(decoder_out, tokens)
            raise ValueError("Обнаружен NaN в лоссе!")

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Ручная оптимизация с проверкой градиентов
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)

        # Проверка градиентов
        self._check_gradients()
        opt.step()

        return loss

    def _check_for_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f"Обнаружен NaN в {name}")

    def _check_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                raise ValueError(f"Обнаружен NaN в градиенте {name}")

    def _log_nan_metrics(self, decoder_out, tokens):
        # Логирование дополнительной информации при NaN
        self.log("nan/decoder_out_mean", decoder_out.mean())
        self.log("nan/tokens_mean", tokens.mean())
        self.log("nan/decoder_out_std", decoder_out.std())

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config['train']['lr'],
        )

        # Добавляем планировщик обучения
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }

fileformer = FileFormer(encoder, decoder, loss_fn, configs)
trainer = L.Trainer(max_epochs=10, precision="16-mixed")
trainer.fit(model=fileformer, train_dataloaders=dataloader)





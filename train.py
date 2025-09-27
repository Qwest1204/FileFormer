from notus import Encoder, Decoder, utils, FileDataset, Muon, ByteLevelTokenizer
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import lightning as L

#tokenizer
tokenizer = ByteLevelTokenizer()

#configs
configs = utils.load_config("/Users/daniilogorodnikov/PycharmProjects/Notus/config/config.yml")

#dataset
dataset = FileDataset(**configs['dataset'])
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
    def __init__(self, encoder, decoder, loss_fn, config, test_loader):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.config = config
        self.test_loader = test_loader
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

        if batch_idx % 500 == 0:
            with torch.no_grad():
                batch = next(iter(self.test_loader))
                tokens, masked_tokens, pads, hash, extention_tokenize = batch

                encoder_out = self.encoder(hash, extention_tokenize)
                decoder_out = self.decoder(masked_tokens, encoder_out, pads)

                print(f"origin tokens : {tokenizer.decode(tokens[:40])}")

                out = torch.argmax(decoder_out, dim=2).tolist()

                print(f"origin tokens : {tokenizer.decode(out[:40])}")

                if configs['train']['temproary_save'] == True:
                    torch.save(self.encoder.state_dict(), "tmp-encoder.pt")
                    torch.save(self.encoder.state_dict(), "tmp-encoder.pt")

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=[*self.encoder.parameters(), *self.decoder.parameters()],
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
trainer = L.Trainer(max_epochs=10, precision=configs['train']['precision'],)
trainer.fit(model=fileformer, train_dataloaders=dataloader)


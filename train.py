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

    def training_step(self, batch, batch_idx):

        tokens, masked_tokens, pads, hash, extention_tokenize = batch

        encoder_out = self.encoder(hash, extention_tokenize)
        decoder_out = self.decoder(masked_tokens, encoder_out, pads)

        loss = loss_fn(decoder_out.view(-1, configs['encoder']['vocab_size']), tokens.view(-1))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['train']['lr'])
        return optimizer

fileformer = FileFormer(encoder, decoder, loss_fn, configs)
trainer = L.Trainer(max_epochs=10, precision="16-mixed")
trainer.fit(model=fileformer, train_dataloaders=dataloader)





import lightning as L
from notus import eval, Encoder, Decoder
import torch.optim as optim

class FileFormer(L.LightningModule):
    def __init__(self, loss_fn, config):
        #comment
        super().__init__()
        self.eval = eval
        self.loss_fn = loss_fn
        self.config = config
        self.automatic_optimization = True  # Для ручного управления оптимизацией

        self.create_models()

    def create_models(self):
        self.encoder = Encoder(**self.config['encoder']).to(self.config['train']['device'])
        self.decoder = Decoder(**self.config['decoder']).to(self.config['train']['device'])

    def training_step(self, batch, batch_idx):
        metadata, tokens, masked_tokens, hash, extention_tokenize = batch
        encoder_out = self.encoder(metadata, hash, extention_tokenize)

        decoder_out = self.decoder(masked_tokens, encoder_out)

        loss = self.loss_fn(
            decoder_out.view(-1, self.config['encoder']['vocab_size']),
            tokens.view(-1)
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        opt = self.optimizers()
        opt.zero_grad()

        opt.step()

        if batch_idx % self.config['train']['interval4save'] == 0:
            self.eval.evaluate(self.forward, batch)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.config['train']['lr'],
        )

        return {
            "optimizer": optimizer,
        }

    def forward(self, meta, data, hash, file_extention, pads=None):
        encoder_out = self.encoder(meta, hash, file_extention)
        decoder_out = self.decoder(data, encoder_out, pads)
        return decoder_out

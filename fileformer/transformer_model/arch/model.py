import lightning as L
from fileformer import eval, Encoder, Decoder
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
        self.encoder = Encoder(**self.config['encoder'])
        self.decoder = Decoder(**self.config['decoder'])

    def training_step(self, batch, batch_idx):
        tokens, masked_tokens, pads, hash, extention_tokenize = batch

        if self.training:  # В training не возвращаем attentions, чтобы не тратить память
            encoder_out = self.encoder(hash, extention_tokenize)
        else:
            encoder_out = self.encoder(hash, extention_tokenize, output_attentions=False)  # Или True если нужно

        decoder_out = self.decoder(masked_tokens, encoder_out, pads)

        loss = self.loss_fn(
            decoder_out.view(-1, self.config['encoder']['vocab_size']),
            tokens.view(-1)
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        opt = self.optimizers()
        opt.zero_grad()

        opt.step()

        if batch_idx % self.configs['train']['interval4save'] == 0:
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

    def forward(self, data, hash, file_extention, pads=None, output_attentions=False):
        if output_attentions:
            encoder_out, encoder_attns = self.encoder(hash, file_extention, output_attentions=True)
            decoder_out, decoder_attns = self.decoder(data, encoder_out, pads, output_attentions=True)
            # Собираем все attentions в один dict или list
            attentions = {"encoder_attentions": encoder_attns, "decoder_attentions": decoder_attns}
            return decoder_out, attentions
        else:
            encoder_out = self.encoder(hash, file_extention)
            decoder_out = self.decoder(data, encoder_out, pads)
            return decoder_out

from fileformer import eval, Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class FileFormer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.create_models()

    def create_models(self, checkpoint=None):
        self.decoder = Decoder(**self.config['decoder'])
        if checkpoint:
            self.decoder.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=self.device))

        return self.decoder


    def training_step(self, batch, batch_idx):
        tokens, pads, _ = batch

        tokens = tokens.to(self.device)
        pads = pads.to(self.device)

        decoder_out = self.decoder(tokens, pads)

        loss = self.loss_fn(
            decoder_out.view(-1, self.config['decoder']['vocab_size']),
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

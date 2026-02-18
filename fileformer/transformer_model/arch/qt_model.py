import lightning as L
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
    QuantStub,
    DeQuantStub,
    float_qparams_weight_only_qconfig,
)
from fileformer import Encoder, Decoder
import torch.optim as optim

class FileFormerQuant(L.LightningModule):
    def __init__(self, loss_fn, config):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn'])
        self.loss_fn = loss_fn
        self.config = config
        self.automatic_optimization = True

        # Создаём FP32 модель
        self.create_models()

        # === QAT SETUP ===
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Подготовка к QAT с установкой qconfig на модулях
        self._setup_qat()

    def create_models(self):
        self.encoder = Encoder(**self.config['encoder'])
        self.decoder = Decoder(**self.config['decoder'])

    def _setup_qat(self):
        # 1. Базовый qconfig для Linear, LayerNorm
        qconfig = get_default_qat_qconfig('fbgemm')  # или 'qnnpack' для MPS

        # 2. Рекурсивно устанавливаем qconfig на подмодулях
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.qconfig = qconfig
            elif isinstance(module, nn.LayerNorm):
                module.qconfig = qconfig
            elif isinstance(module, nn.Embedding):
                # Weight-only для Embedding (опционально; можно None для FP32)
                module.qconfig = float_qparams_weight_only_qconfig
            else:
                # Для custom (RotaryPE, Attention) — None, чтобы не квантизовать
                module.qconfig = None

        # 3. Применяем prepare_qat (без qconfig_dict)
        self.model_prepared = prepare_qat(self, inplace=False)

    def forward(self, data, hash, file_extention, pads=None, output_attentions=False):
        # Вход в int8 (опционально)
        #x = self.quant(masked_tokens.float())
        # Используем encoder/decoder
        encoder_out = self.encoder(hash, file_extention)
        decoder_out = self.decoder(data, encoder_out, pads)
        return self.dequant(decoder_out)

    def training_step(self, batch, batch_idx):
        tokens, masked_tokens, pads, hash, extention_tokenize = batch

        decoder_out = self.model_prepared.forward(
            masked_tokens, hash, extention_tokenize, pads
        )

        loss = self.loss_fn(
            decoder_out.view(-1, self.config['encoder']['vocab_size']),
            tokens.view(-1)
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_end(self):
        print("Converting to INT8...")
        # Move to CPU before convert, as quantized ops are CPU-only
        self.model_prepared = self.model_prepared.cpu()
        self.int8_model = convert(self.model_prepared)
        torch.save(self.int8_model.state_dict(), "fileformer_int8_final.pth")
        print("INT8 model saved!")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model_prepared.parameters(),
            lr=self.config['train']['lr'] * 0.1,
        )
        return optimizer
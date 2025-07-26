from notus import EncoderDecoder, Encoder, Decoder
from notus import ByteLevelTokenizer, FileDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

encoder = Encoder()
decoder = Decoder()
trainning_model = EncoderDecoder(encoder=encoder, decoder=decoder)

dataset = FileDataset("path", max_seq_length=512, tokenizer=ByteLevelTokenizer())
dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=dataset.collate_fn
    )

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="cuda",
    precision=16
)

trainer.fit(trainning_model, dataloader)

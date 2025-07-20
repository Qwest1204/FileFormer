from notus.transformer_model import FileTransformerBlock, ValueTransformerBlock, FileTransformer
from notus import ByteLevelTokenizer, FileDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

transformer = FileTransformerBlock(max_seq_len=512, embedding_tensor=512, num_encoder_layers=2, nhead=2)
transformerv = ValueTransformerBlock(max_seq_len=512, correction_tensor=512, num_encoder_layers=2, nhead=2)
trainning_model = FileTransformer(transformerv, transformer, 1)

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

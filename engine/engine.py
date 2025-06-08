from Notus import Tokenizer, Transformer
import torch

class CompressEngine:
    def __init__(self, tokenizer: Tokenizer, transformer: Transformer, chunk_size=256):
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.chunk_size = chunk_size

    def compress(self, hex):
        tokens = self.tokenizer.encode(hex)
        pad_token = self.tokenizer.get_special_token_id("<|PAD|>")

        chunks = []
        masks = []

        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            padding_count = self.chunk_size - len(chunk)

            if padding_count > 0:
                chunk = chunk + [pad_token] * padding_count

            mask = [1] * len(chunk)
            if padding_count > 0:
                mask[-padding_count:] = [0] * padding_count

            chunks.append(chunk)
            masks.append(mask)

        compress_data = []
        for chunk, mask in zip(torch.tensor(chunks, dtype=torch.int32), torch.tensor(masks, dtype=torch.int32)):
            compress_data.append(self.transformer.encode(chunk, mask))

        return compress_data, torch.tensor(masks, dtype=torch.int32)

    def decompress(self, data, mask):
        hex_d = []
        for z, mask in zip(data, mask):
            x = self.transformer.project(self.transformer.decode(z, mask.unsqueeze(0).unsqueeze(0)))
            hex_d.extend(torch.max(x.view(-1, x.shape[-1]), dim=1)[1].tolist())

        bites = bytes.fromhex(self.tokenizer.decode(hex_d))
        return bites


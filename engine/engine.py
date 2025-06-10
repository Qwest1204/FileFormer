from Notus import Tokenizer, Transformer
import torch
from tqdm import tqdm

class CompressEngine:
    def __init__(self, tokenizer: Tokenizer, transformer: Transformer, chunk_size=256):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = transformer.to(self.device)

    def compress(self, hex_str: str):
        tokens = self.tokenizer.encode(hex_str)
        pad_token = self.tokenizer.get_special_token_id("<|PAD|>")
        num_chunks = (len(tokens) + self.chunk_size - 1) // self.chunk_size

        # Инициализация тензоров с заполнением паддингом
        chunks = torch.full((num_chunks, self.chunk_size), pad_token, dtype=torch.int32).to(self.device)
        masks = torch.zeros((num_chunks, self.chunk_size), dtype=torch.int32).to(self.device)

        # Заполнение действительными токенами и масками
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            actual_len = min(end, len(tokens)) - start
            chunk_data = tokens[start:end]

            chunks[i, :actual_len] = torch.tensor(chunk_data, dtype=torch.int32).to(self.device)
            masks[i, :actual_len] = 1

        # Пакетное сжатие всех чанков
        compress_data = [self.transformer.encode(chunk, mask).to(self.device) for chunk, mask in tqdm(zip(chunks, masks))]
        return compress_data, masks

    def decompress(self, data, masks) -> bytes:
        # Проверяем поддержку пакетной обработки
        try:
            # Пытаемся обработать весь батч
            z_batch = torch.stack(data).to(self.device)
            logits = self.transformer.project(
                self.transformer.decode(z_batch, masks.unsqueeze(0).unsqueeze(0).to(self.device))).to(self.device)
            token_ids = torch.argmax(logits, dim=-1).view(-1)
        except (RuntimeError, TypeError):
            # Фолбэк на последовательную обработку
            token_ids = []
            for z, mask in tqdm(zip(data, masks)):
                logits = self.transformer.project(
                    self.transformer.decode(z, mask.unsqueeze(0)).to(self.device)).to(self.device)
                token_ids.extend(torch.argmax(logits, dim=-1))
            token_ids = torch.cat(token_ids).to(self.device)

        hex_str = self.tokenizer.decode(token_ids.tolist())

        if len(hex_str) % 2 == 0:
            return bytes.fromhex(hex_str)
        else:
            return bytes.fromhex(hex_str[:-1])

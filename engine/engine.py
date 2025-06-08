from Notus import Tokenizer, Transformer
import torch


class CompressEngine:
    def __init__(self, tokenizer: Tokenizer, transformer: Transformer, chunk_size=256):
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.chunk_size = chunk_size

    def compress(self, hex_str: str):
        tokens = self.tokenizer.encode(hex_str)
        pad_token = self.tokenizer.get_special_token_id("<|PAD|>")
        num_chunks = (len(tokens) + self.chunk_size - 1) // self.chunk_size

        # Инициализация тензоров с заполнением паддингом
        chunks = torch.full((num_chunks, self.chunk_size), pad_token, dtype=torch.int32)
        masks = torch.zeros((num_chunks, self.chunk_size), dtype=torch.int32)

        # Заполнение действительными токенами и масками
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            actual_len = min(end, len(tokens)) - start
            chunk_data = tokens[start:end]

            chunks[i, :actual_len] = torch.tensor(chunk_data, dtype=torch.int32)
            masks[i, :actual_len] = 1

        # Пакетное сжатие всех чанков
        compress_data = [self.transformer.encode(chunk, mask) for chunk, mask in zip(chunks, masks)]
        return compress_data, masks

    def decompress(self, data, masks):
        # Проверяем поддержку пакетной обработки
        try:
            # Пытаемся обработать весь батч
            z_batch = torch.stack(data)
            logits = self.transformer.project(
                self.transformer.decode(z_batch, masks.unsqueeze(0).unsqueeze(0)))
            token_ids = torch.argmax(logits, dim=-1).view(-1)
        except (RuntimeError, TypeError):
            # Фолбэк на последовательную обработку
            token_ids = []
            for z, mask in zip(data, masks):
                logits = self.transformer.project(
                    self.transformer.decode(z, mask.unsqueeze(0)))
                token_ids.extend(torch.argmax(logits, dim=-1))
            token_ids = torch.cat(token_ids)

        hex_str = self.tokenizer.decode(token_ids.tolist())
        return hex_str#bytes.fromhex(hex_str)
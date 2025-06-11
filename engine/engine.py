from Notus import Tokenizer
from Notus import Transformer
import torch
from tqdm import tqdm

class CompressEngine:
    def __init__(self, tokenizer: Tokenizer, compress_unit: Transformer,
                 error_fix_unit: Transformer, chunk_compress_size=256,):
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_token = self.tokenizer.get_special_token_id("<|PAD|>")
        self.mask_token = self.tokenizer.get_special_token_id("<|UNK|>")
        self.compress_unit = compress_unit.to(self.device)
        self.error_fix_unit = error_fix_unit.to(self.device)
        self.chunk_compress_size = chunk_compress_size

    def apply_mask(self, tensor, mask):
        # Создаем копию исходного тензора
        new_tensor = tensor.clone()

        # Применяем маску через булеву индексацию
        new_tensor[mask.bool()] = self.mask_token

        return new_tensor

    def max_probabilities(self, tensor1, tensor2):
        """
        Для каждого из 256 элементов выбирает значение с максимальной вероятностью между двумя тензорами,
        возвращает индексы выбранных значений.

        Args:
            tensor1 (torch.Tensor): Первый тензор размерности [256, 5667]
            tensor2 (torch.Tensor): Второй тензор размерности [256, 5667]

        Returns:
            torch.Tensor: Тензор индексов выбранных значений размерности [256]
        """
        # Находим максимальные значения и их индексы для каждого тензора
        max_vals1, max_indices1 = torch.max(tensor1, dim=-1)
        max_vals2, max_indices2 = torch.max(tensor2, dim=-1)

        # Сравниваем максимальные значения и выбираем источник
        source_mask = max_vals1 >= max_vals2

        # Формируем итоговые индексы
        final_indices = torch.where(source_mask, max_indices1, max_indices2)
        return final_indices

    def create_mask(self, tensor1, tensor2):
        initial_mask = (tensor1 != tensor2).int()
        new_mask = initial_mask.clone()

        # Находим индексы единиц в начальной маске
        indices = torch.where(initial_mask == 1)[0]

        # Если есть хотя бы одна единица
        if indices.numel() > 0:
            # Вычисляем индексы левых и правых соседей
            left_indices = indices - 1
            right_indices = indices + 1

            # Фильтруем индексы, чтобы не выйти за границы
            left_indices = left_indices[left_indices >= 0]
            right_indices = right_indices[right_indices < initial_mask.size(0)]

            # Объединяем все индексы соседей
            all_indices = torch.cat([left_indices, right_indices])

            # Устанавливаем 1 для всех соседних позиций
            new_mask[all_indices] = 1

        return new_mask

    def compress(self, data_from_file: bytes):
        tokens = self.tokenizer.encode(data_from_file.hex())

        num_chunks = (len(tokens) + self.chunk_compress_size - 1) // self.chunk_compress_size

        chunks = torch.full((num_chunks, self.chunk_compress_size), self.pad_token, dtype=torch.int32).to(self.device)
        masks = torch.zeros((num_chunks, self.chunk_compress_size), dtype=torch.int32).to(self.device)

        for i in range(num_chunks):
            start = i * self.chunk_compress_size
            end = start + self.chunk_compress_size
            actual_len = min(end, len(tokens)) - start
            chunk_data = tokens[start:end]

            chunks[i, :actual_len] = torch.tensor(chunk_data, dtype=torch.int32).to(self.device)
            masks[i, :actual_len] = 1

        compress_data = [self.compress_unit.encode(chunk, mask).to(self.device) for chunk, mask in tqdm(zip(chunks, masks))]
        return compress_data, masks

    def decompress(self, data, mask, deep_of_error_correction) -> bytes:
        token_ids = []
        for data, mask in tqdm(zip(data, mask)):
            decoder_out_main = self.compress_unit.project(
                self.compress_unit.decode(data, mask.unsqueeze(0)).to(self.device)
            ).to(self.device)
            decoder_out_main = torch.argmax(decoder_out_main, dim=-1)

            decoder_out_alternate = self.compress_unit.project(
                self.compress_unit.decode(data, mask.unsqueeze(0)).to(self.device)
            ).to(self.device)
            decoder_out_alternate = torch.argmax(decoder_out_alternate, dim=-1)

            for _ in range(deep_of_error_correction):
                mask_mask = self.create_mask(decoder_out_main, decoder_out_alternate)
                mark_src = self.apply_mask(decoder_out_main, mask_mask)

                decoder_out_main_n = self.error_fix_unit.forward(mark_src, mask)
                decoder_out_alternate_n = self.error_fix_unit.forward(mark_src, mask)

                decoder_out_main = torch.argmax(decoder_out_main_n, dim=-1)
                decoder_out_alternate = torch.argmax(decoder_out_alternate_n, dim=-1)

            token_ids.extend(self.max_probabilities(decoder_out_main_n, decoder_out_alternate_n))
        token_ids = torch.cat(token_ids).to(self.device)
        hex_str = self.tokenizer.decode(token_ids.tolist())

        if len(hex_str) % 2 == 0:
            return bytes.fromhex(hex_str)
        else:
            return bytes.fromhex(hex_str[:-1])

import torch
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import hashlib
from notus.tokenizer import ByteLevelTokenizer
import pandas as pd
from notus.tokenizer.utils import get_file_ext_as_token
import numpy as np

class FileDataset(Dataset):
    def __init__(self, path, seq_len, vocab_size):
        self.files = [x for x in Path(path).glob('**/*') if x.is_file() and x.stat().st_size > 0]  # Исключаем пустые файлы
        self.tokenizer = ByteLevelTokenizer()
        self.seq_len = seq_len
        self.vocab_size = vocab_size  # Добавляем проверку размера словаря
        self.pad_token = self.tokenizer.encode("<pad>")[0]
        self.mask_token = self.tokenizer.encode("<mask>")[0]
        self.table_init_data = pd.DataFrame(columns=["id", "file_name", "start_byte", "end_byte"])

    def prepare(self):
        byte_len = 0
        for i, file_path in enumerate(self.files):
            try:
                file_size = file_path.stat().st_size
                if file_size == 0:
                    continue  # Пропускаем пустые файлы

                num_chunks = (file_size + self.seq_len - 1) // self.seq_len  # Корректное вычисление чанков
                new_data = pd.DataFrame({
                    'id': [i],
                    'file_name': [str(file_path)],
                    'start_byte': [byte_len],
                    'end_byte': [byte_len + num_chunks - 1]
                })
                self.table_init_data = pd.concat([self.table_init_data, new_data], ignore_index=True)
                byte_len += num_chunks
            except Exception as e:
                print(f"Ошибка обработки файла {file_path}: {e}")
                continue

    def get_file_name_by_byte(self, byte_value):
        try:
            row = self.table_init_data[
                (self.table_init_data['start_byte'] <= byte_value) &
                (self.table_init_data['end_byte'] >= byte_value)
                ].iloc[0]
            return row['file_name'], row['start_byte'], row['end_byte']
        except IndexError:
            return None

    def read_file(self, file_name, start_pos):
        try:
            with open(file_name, 'rb') as f:
                f.seek(start_pos * self.seq_len)  # Умножаем на seq_len для получения байтовой позиции
                data = f.read(self.seq_len)

                if len(data) == 0:
                    return torch.full((self.seq_len,), self.pad_token), torch.zeros((1, self.seq_len))

                # Конвертируем байты в hex строку для токенизации
                hex_data = data.hex()
                tokens = self.tokenizer.encode(hex_data)
                #print(tokens)
                # Обрезаем или дополняем до seq_len
                if len(tokens) > self.seq_len:
                    tokens = tokens[:self.seq_len]
                    padding_mask = torch.ones((1, self.seq_len))
                else:
                    padding_mask = torch.cat([
                        torch.ones((1, len(tokens))),
                        torch.zeros((1, self.seq_len - len(tokens)))
                    ], dim=1)
                    tokens.extend([self.pad_token] * (self.seq_len - len(tokens)))

                # Проверяем валидность токенов
                #tokens = [t if t < self.vocab_size else self.pad_token for t in tokens]
                return torch.tensor(tokens, dtype=torch.long), padding_mask

        except Exception as e:
            print(f"Ошибка чтения файла {file_name}: {e}")
            return torch.full((self.seq_len,), self.pad_token), torch.zeros((1, self.seq_len))

    def mask_tokens(self, x):
        masked = x.clone()
        if random.random() > 0.6:
            return masked

        # Маскируем только не-pad токены
        non_pad_indices = (masked != self.pad_token).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            mask_indices = random.sample(non_pad_indices.tolist(),
                                         k=max(1, int(len(non_pad_indices) * 0.15)))
            masked[mask_indices] = self.mask_token

        return masked

    def get_hesh_tokens(self, file_name):
        try:
            sha256_hash = hashlib.sha256()
            with open(file_name, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            hex_hash = sha256_hash.hexdigest()
            hash_tokens = self.tokenizer.encode(hex_hash)

            # Фиксированная длина для хеша - 64 токена
            if len(hash_tokens) > 64:
                hash_tokens = hash_tokens[:64]  # Обрезаем если слишком длинный
            else:
                # Дополняем до 64 токенов
                hash_tokens.extend([self.pad_token] * (64 - len(hash_tokens)))

            return torch.tensor(hash_tokens, dtype=torch.long)

        except Exception as e:
            print(f"Ошибка вычисления хеша {file_name}: {e}")
            return torch.tensor([self.pad_token] * 64, dtype=torch.long)  # Всегда возвращаем 64 токена  # SHA256 всегда имеет длину 64 символа

    def __len__(self):
        return int(self.table_init_data['end_byte'].max() + 1) if not self.table_init_data.empty else 0

    def __getitem__(self, index):
        try:
            info = self.get_file_name_by_byte(index)
            if info is None:
                # Возвращаем пустые данные если файл не найден
                empty_tokens = torch.full((self.seq_len,), self.pad_token)
                empty_mask = torch.zeros((1, self.seq_len))
                empty_hash = torch.tensor([self.pad_token] * 64, dtype=torch.long)
                empty_ext = torch.tensor([0], dtype=torch.long)
                return empty_tokens, empty_tokens.clone(), empty_mask, empty_hash, empty_ext

            file_name, start_byte, end_byte = info
            chunk_index = index - start_byte  # Вычисляем индекс чанка в файле

            tokens, pads = self.read_file(file_name, chunk_index)
            masked_tokens = self.mask_tokens(tokens)
            hash = self.get_hesh_tokens(file_name)
            extention_tokenize = torch.tensor([get_file_ext_as_token(file_name)], dtype=torch.long)

            #Финальная проверка на NaN и бесконечные значения
            if torch.isnan(tokens).any() or torch.isinf(tokens).any():
               print(f"Обнаружены NaN/Inf в токенах файла {file_name}")
               tokens = torch.where(torch.isnan(tokens) | torch.isinf(tokens),
                                    torch.tensor(self.pad_token), tokens)

            return tokens, masked_tokens, pads, hash, extention_tokenize

        except Exception as e:
            print(f"Error in __getitem__ for index {index}: {e}")
            empty_tokens = torch.full((self.seq_len,), self.pad_token)
            empty_mask = torch.zeros((1, self.seq_len))
            empty_hash = torch.tensor([self.pad_token] * 64, dtype=torch.long)
            empty_ext = torch.tensor([0], dtype=torch.long)
            return empty_tokens, empty_tokens.clone(), empty_mask, empty_hash, empty_ext
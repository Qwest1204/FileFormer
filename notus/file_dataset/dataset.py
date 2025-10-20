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
import bisect
import mmap
import os

class FileDataset(Dataset):
    def __init__(self, path, seq_len, vocab_size):
        self.files = [x for x in Path(path).glob('**/*') if x.is_file() and x.stat().st_size > 0]  # Исключаем пустые файлы
        self.tokenizer = ByteLevelTokenizer()
        self.seq_len = seq_len
        self.vocab_size = vocab_size  # Добавляем проверку размера словаря
        self.pad_token = self.tokenizer.encode("<pad>")[0]
        self.mask_token = self.tokenizer.encode("<mask>")[0]
        self.file_infos = []
        self.chunk_starts = []  # Для bisect поиска
        self.total_chunks = 0
        self.open_files = {}  # Для хранения открытых файлов и mmap
        self.prepare()

    def prepare(self):
        chunk_offset = 0
        data_list = []  # Собираем данные в список для эффективного создания DataFrame позже, если нужно

        for file_path in tqdm(self.files, desc="Preparing files"):
            file_name = str(file_path)
            try:
                file_size = file_path.stat().st_size
                if file_size == 0:
                    continue

                num_chunks = (file_size + self.seq_len - 1) // self.seq_len

                # Вычисляем хеш один раз
                sha256_hash = hashlib.sha256()
                with open(file_name, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                hex_hash = sha256_hash.hexdigest()
                hash_tokens = self.tokenizer.encode(hex_hash)
                if len(hash_tokens) > 64:
                    hash_tokens = hash_tokens[:64]
                else:
                    hash_tokens.extend([self.pad_token] * (64 - len(hash_tokens)))
                hash_tensor = torch.tensor(hash_tokens, dtype=torch.long)

                # Токен расширения файла
                ext_token = torch.tensor([get_file_ext_as_token(file_name)], dtype=torch.long)
                with open(file_path, 'rb') as f:
                    meta = f.read(1024).hex()

                # Открываем файл и создаем mmap для быстрого доступа
                fd = open(file_name, 'rb')
                mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
                self.open_files[file_name] = (fd, mm)

                # Добавляем информацию о файле
                self.file_infos.append({
                    'file_name': file_name,
                    'metadata': meta,
                    'start_chunk': chunk_offset,
                    'end_chunk': chunk_offset + num_chunks - 1,
                    'hash_tokens': hash_tensor,
                    'ext_token': ext_token,
                    'mmap': mm
                })
                self.chunk_starts.append(chunk_offset)

                chunk_offset += num_chunks

            except Exception as e:
                print(f"Ошибка обработки файла {file_path}: {e}")
                continue

        self.total_chunks = chunk_offset
        # Если нужно, можно создать DataFrame здесь: self.table_init_data = pd.DataFrame(data_list)

    def get_file_info(self, index):
        # Используем bisect для быстрого поиска файла по индексу чанка
        i = bisect.bisect_left(self.chunk_starts, index + 1) - 1
        if i < 0:
            i = 0
        if i >= len(self.file_infos):
            return None
        info = self.file_infos[i]
        if info['start_chunk'] <= index <= info['end_chunk']:
            return info
        return None

    def read_file(self, info, chunk_index):
        try:
            mm = info['mmap']
            start_byte = chunk_index * self.seq_len
            data = mm[start_byte: start_byte + self.seq_len]

            if len(data) == 0:
                return torch.full((self.seq_len,), self.pad_token), torch.zeros((1, self.seq_len))

            # Конвертируем байты в hex строку для токенизации
            hex_data = data.hex()
            tokens = self.tokenizer.encode(hex_data)

            # Обрезаем или дополняем до seq_len
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
                padding_mask = torch.ones((1, self.seq_len))
            else:
                padding_mask = torch.cat([
                    torch.ones((1, len(tokens))),
                    torch.zeros((1, self.seq_len - len(tokens)))
                ], dim=1)
                tokens += [self.pad_token] * (self.seq_len - len(tokens))

            return torch.tensor(tokens, dtype=torch.long), padding_mask

        except Exception as e:
            print(f"Ошибка чтения файла {info['file_name']}: {e}")
            return torch.full((self.seq_len,), self.pad_token), torch.zeros((1, self.seq_len))

    def mask_tokens(self, x):
        masked = x.clone()
        if random.random() > 0.6:
            return masked

        # Маскируем только не-pad токены
        non_pad_indices = (masked != self.pad_token).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) == 0:
            return masked

        num_to_mask = max(1, int(len(non_pad_indices) * 0.15))
        mask_indices = random.sample(non_pad_indices.tolist(), k=num_to_mask)
        masked[mask_indices] = self.mask_token

        return masked

    def processing_meta(self, index):
        meta = self.get_file_info(index)['metadata']
        tokens = self.tokenizer.encode(meta)
        return torch.tensor([tokens], dtype=torch.long)

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, index):
        info = self.get_file_info(index)
        if info is None:
            empty_tokens = torch.full((self.seq_len,), self.pad_token)
            empty_mask = torch.zeros((1, self.seq_len))
            empty_hash = torch.tensor([self.pad_token] * 64, dtype=torch.long)
            empty_ext = torch.tensor([0], dtype=torch.long)
            return empty_tokens, empty_tokens.clone(), empty_mask, empty_hash, empty_ext

        chunk_index = index - info['start_chunk']
        metatokens = self.processing_meta(index)
        tokens, pads = self.read_file(info, chunk_index)
        masked_tokens = self.mask_tokens(tokens)

        # Финальная проверка на NaN и бесконечные значения (хотя для токенов это маловероятно)
        if torch.isnan(tokens).any() or torch.isinf(tokens).any():
            print(f"Обнаружены NaN/Inf в токенах файла {info['file_name']}")
            tokens = torch.where(torch.isnan(tokens) | torch.isinf(tokens), torch.tensor(self.pad_token), tokens)

        return metatokens, tokens, masked_tokens, pads, info['hash_tokens'], info['ext_token']

    def __del__(self):
        # Закрываем все открытые файлы и mmap при уничтожении объекта
        for fd, mm in self.open_files.values():
            mm.close()
            fd.close()
import random
import torch
import re
from typing import Tuple
import os
import glob

from torch.utils.data import Dataset
from safetensors.torch import load_file
from fileformer.tokenizer import ByteLevelTokenizer


class FileDataset(Dataset):
    def __init__(self, path:str, ratio:int):
        """
        path: корневая директория, содержащая подпапки с чанками данных.
        """
        self.path = path
        self.tokenizer = ByteLevelTokenizer()
        self.pad_token_id = self.tokenizer.encode("<pad>")[0]
        self.mask_token_id = self.tokenizer.encode("<mask>")[0]
        self.vocab = self.tokenizer.vocab_size
        self.ratio = ratio

        # Получаем все поддиректории (каждая соответствует одному исходному файлу)
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        subdirs.sort()
        self.subdirs = subdirs

        # Построение индексной карты: для каждой папки храним путь, число чанков и начальный индекс
        self.folders_info = []
        total_chunks = 0

        for subdir in subdirs:
            folder_path = os.path.join(path, subdir)

            # Считаем все файлы data*.safetensors в папке
            data_files = glob.glob(os.path.join(folder_path, "data*.safetensors"))
            num_chunks = len(data_files)
            if num_chunks == 0:
                # Папка без чанков – пропускаем (можно логировать предупреждение)
                continue

            # Проверяем наличие файла метаданных
            meta_path = os.path.join(folder_path, "meta.safetensors")
            if not os.path.isfile(meta_path):
                # Нет метаданных – пропускаем (хотя по заданию они должны быть)
                continue

            self.folders_info.append({
                'path': folder_path,
                'num_chunks': num_chunks,
                'start_idx': total_chunks
            })
            total_chunks += num_chunks

        self.total_chunks = total_chunks
        self.metadata_cache = {}  # кэш для загруженных метаданных

    def mask_tokens(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rand_vals = torch.rand_like(x, dtype=torch.float)
        # Создаём булеву маску: True с вероятностью self.ratio (токены, которые заменим)
        mask = rand_vals < self.ratio
        # Исключаем pad-токены из маски
        mask = mask & (x != self.pad_token_id)
        # Заменяем отмеченные токены на mask_token_id, остальные оставляем без изменений
        masked_x = torch.where(mask, self.mask_token_id, x)
        return masked_x, mask

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError(f"Index {idx} out of range [0, {self.total_chunks})")

        # Находим папку, содержащую запрошенный индекс
        folder_info = None
        for info in self.folders_info:
            if idx < info['start_idx'] + info['num_chunks']:
                folder_info = info
                break
        if folder_info is None:
            raise RuntimeError("Index not found in folders_info")  # не должно случаться

        local_idx = idx - folder_info['start_idx']
        folder_path = folder_info['path']

        # Загружаем файл с данными
        data_file = os.path.join(folder_path, f"data{local_idx}.safetensors")
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"Missing data file: {data_file}")
        data_tensors = load_file(data_file)

        # Загружаем (или достаём из кэша) метаданные
        meta_file = os.path.join(folder_path, "meta.safetensors")
        if meta_file not in self.metadata_cache:
            if not os.path.isfile(meta_file):
                raise FileNotFoundError(f"Missing metadata file: {meta_file}")
            self.metadata_cache[meta_file] = load_file(meta_file)
        meta_tensors = self.metadata_cache[meta_file]

        masked_data, mask = self.mask_tokens(data_tensors['tokenized_data'])

        # Возвращаем словарь с тензорами токенов и хешами (хеши опциональны)
        return masked_data, mask, data_tensors['tokenized_data'], meta_tensors['tokenized_metadata'], data_tensors.get('hash_tokens'), meta_tensors.get('hash_tokens')
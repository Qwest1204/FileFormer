import random
import torch
import re
from typing import Tuple
import os
import glob
import hashlib
import json

from torch.utils.data import Dataset
from safetensors.torch import load_file, safe_open, save_file
from fileformer.tokenizer import ByteLevelTokenizer


class FileDataset(Dataset):
    def __init__(self, path:str, ratio:float):
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

        # Возвращаем словарь с тензорами токенов и хешами (хеши опциональны)
        return data_tensors['tokenized_data'], meta_tensors['tokenized_metadata'], data_tensors.get('hash_tokens'), meta_tensors.get('hash_tokens')


class ENWIK8Dataset(Dataset):
    """
       Args:
           file_path (str): путь к бинарному файлу.
           tokenizer: токенизатор с методами encode и pad_token_id.
           seq_len (int): желаемая длина последовательности токенов.
           overlap (int): количество перекрывающихся токенов между соседними окнами.
           cache_dir (str, optional): директория для сохранения кэша. Если None,
                                       используется папка рядом с file_path.
           force_rebuild (bool): принудительно пересоздать кэш, даже если он существует.
       """

    def __init__(self, file_path:str, tokenizer:ByteLevelTokenizer, seq_len:int, overlap:int, cache_dir=None, force_rebuild=False):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.overlap = overlap
        self.stride = seq_len - overlap

        if self.stride <= 0:
            raise ValueError("overlap должно быть меньше seq_len")

        # Определяем путь для кэша
        if cache_dir is None:
            cache_dir = os.path.dirname(file_path)
        os.makedirs(cache_dir, exist_ok=True)

        # Генерируем уникальное имя для кэша на основе параметров
        params = f"{os.path.basename(file_path)}_{seq_len}_{overlap}_{os.path.getsize(file_path)}"
        hash_id = hashlib.md5(params.encode()).hexdigest()

        self.cache_path = os.path.join(cache_dir, f"hexds_{hash_id}.safetensors")
        self.meta_path = os.path.join(cache_dir, f"hexds_{hash_id}.json")

        if not force_rebuild and os.path.exists(self.cache_path) and os.path.exists(self.meta_path):
            # Загружаем метаданные из JSON
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
            self.num_samples = meta["num_samples"]
            self.seq_len = meta["seq_len"]
            self.overlap = meta["overlap"]
        else:
            # Кэша нет — строим датасет с нуля
            self._build_cache(file_path)

        # Открываем safetensors-файл для последующего чтения по индексу
        # Объект остаётся открытым на всё время жизни датасета
        self.safetensors = safe_open(self.cache_path, framework="pt", device="cpu")

    def _build_cache(self, file_path):
        """Строит кэш: читает файл, токенизирует, создаёт окна и сохраняет их в safetensors."""
        # Чтение всего файла (всё равно необходимо для токенизации)
        with open(file_path, 'rb') as f:
            byte_data = f.read()
        hex_str = byte_data.hex()
        full_tokens = self.tokenizer.encode(hex_str)  # полный список токенов

        # Разбиение на окна
        samples = []
        total_len = len(full_tokens)
        start = 0
        while start + self.seq_len <= total_len:
            chunk = full_tokens[start:start + self.seq_len]
            samples.append(chunk)
            start += self.stride

        # Последнее неполное окно
        if start < total_len:
            chunk = full_tokens[start:]
            pad_len = self.seq_len - len(chunk)
            chunk = chunk + [self.tokenizer.encode("<pad>")[0]] * pad_len
            samples.append(chunk)

        # Преобразуем в тензоры и создаём словарь для safetensors
        tensor_dict = {str(i): torch.tensor(seq, dtype=torch.long) for i, seq in enumerate(samples)}
        self.num_samples = len(samples)

        # Сохраняем safetensors
        save_file(tensor_dict, self.cache_path)

        # Сохраняем метаданные
        meta = {
            "num_samples": self.num_samples,
            "seq_len": self.seq_len,
            "overlap": self.overlap
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

        # Очищаем большие списки (они больше не нужны)
        del full_tokens, samples, tensor_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Загружаем только один тензор по ключу (индекс как строка)
        return self.safetensors.get_tensor(str(idx))

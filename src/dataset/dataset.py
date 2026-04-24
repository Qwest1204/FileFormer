import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch
from typing import List, Optional, Union
from tokenizer import ByteLevelTokenizer

class MultiFileDataset(Dataset):
    """
    Датасет для GPT-обучения на текстах/бинарных файлах с байтовым токенизатором.

    Параметры
    ----------
    folder_path : str or Path
        Путь к корневой папке с данными.
    extensions : List[str]
        Список расширений файлов (например, ['.txt', '.py']) без точки.
    seq_len : int
        Длина одной последовательности в токенах (без учёта сдвига).
    cache_path : str or Path
        Путь к файлу .npy для сохранения/загрузки обработанных данных.
    tokenizer : ByteLevelTokenizer, optional
        Экземпляр токенизатора. Если не передан, создаётся стандартный.
    """
    def __init__(
        self,
        folder_path: Union[str, Path],
        extensions: List[str],
        seq_len: int,
        cache_path: Union[str, Path],
        tokenizer: Optional['ByteLevelTokenizer'] = None
    ):
        self.seq_len = seq_len
        self.cache_path = Path(cache_path)
        self.tokenizer = tokenizer or ByteLevelTokenizer()

        # Если кэш существует – загружаем, иначе создаём
        if self.cache_path.exists():
            self.data = np.load(str(self.cache_path), mmap_mode='r')
        else:
            self.data = self._build_cache(folder_path, extensions)
            np.save(str(self.cache_path), self.data)
            # Открываем для чтения с memory-mapping
            self.data = np.load(str(self.cache_path), mmap_mode='r')

    def _build_cache(self, folder_path: Union[str, Path], extensions: List[str]) -> np.ndarray:
        """Обход папки, токенизация, объединение и нарезка на блоки (seq_len + 1)."""
        folder = Path(folder_path)
        all_files = []
        for ext in extensions:
            all_files.extend(folder.rglob(f'*.{ext}'))
        if not all_files:
            raise FileNotFoundError(f'Нет файлов с расширениями {extensions} в {folder}')

        # Токенизируем все файлы и конкатенируем токены
        all_tokens = []
        for file_path in sorted(all_files):
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            hex_str = raw_bytes.hex()          # преобразуем байты в hex-строку
            tokens = self.tokenizer.encode(hex_str)
            all_tokens.extend(tokens)

        if len(all_tokens) == 0:
            raise ValueError('После токенизации не получено ни одного токена')

        # Нарезка на непересекающиеся блоки размера seq_len + 1.
        # Каждый блок даёт src = block[:-1], tgt = block[1:]
        block_size = self.seq_len + 1
        num_blocks = len(all_tokens) // block_size
        usable_tokens = all_tokens[:num_blocks * block_size]
        blocks = np.array(usable_tokens, dtype=np.int32).reshape(num_blocks, block_size)
        return blocks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        block = self.data[idx]                # массив длиной seq_len + 1
        src = torch.tensor(block[:-1].astype(np.int64), dtype=torch.long)   # int64 для PyTorch
        tgt = torch.tensor(block[1:].astype(np.int64), dtype=torch.long)
        return src, tgt
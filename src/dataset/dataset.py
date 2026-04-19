import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
from safetensors.torch import save_file
from safetensors import safe_open
import hashlib
from model import ByteLevelTokenizer


class FileDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: int,
        mask_prob: float = 0.2,          # доля предсказываемых токенов (20%)
        cache_dir: str = None,
        force_rebuild: bool = False,
    ):
        self.tokenizer = ByteLevelTokenizer()
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.mask_token_id = self.tokenizer.encode("<MASK>")[0]

        if cache_dir is None:
            cache_dir = os.path.dirname(file_path)
        os.makedirs(cache_dir, exist_ok=True)

        # Уникальный хэш кэша
        params = (
            f"{os.path.basename(file_path)}_{seq_len}_"
            f"{mask_prob}_{os.path.getsize(file_path)}"
        )
        hash_id = hashlib.md5(params.encode()).hexdigest()
        self.cache_path = os.path.join(cache_dir, f"hexds_{hash_id}.safetensors")

        if not force_rebuild and os.path.exists(self.cache_path):
            self.safetensors = safe_open(self.cache_path, framework="pt", device="cpu")
            self.num_samples = len([k for k in self.safetensors.keys() if k.startswith("input_ids_")])
        else:
            self._build_cache(file_path)

    def _build_cache(self, file_path):
        with open(file_path, 'rb') as f:
            byte_data = f.read()
        hex_str = byte_data.hex()
        full_tokens = self.tokenizer.encode(hex_str)

        samples = []
        total_len = len(full_tokens)
        start = 0

        while start + self.seq_len <= total_len:
            chunk = full_tokens[start:start + self.seq_len]
            samples.append(chunk)
            start += self.seq_len

        tensor_dict = {}
        for i, seq in enumerate(samples):
            tensor_dict[f"input_ids_{i}"] = torch.tensor(seq, dtype=torch.long)

        self.num_samples = len(samples)
        save_file(tensor_dict, self.cache_path)
        self.safetensors = safe_open(self.cache_path, framework="pt", device="cpu")

        del full_tokens, samples, tensor_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Исходная последовательность
        original = self.safetensors.get_tensor(f"input_ids_{idx}").clone()
        src_seq = original.clone()
        tgt_seq = torch.full_like(original, -100)  # по умолчанию игнорируем всё

        # Выбираем токены для предсказания (mask_prob)
        prob = torch.rand(original.shape)
        selected = prob < self.mask_prob          # маска предсказываемых позиций
        num_selected = selected.sum().item()

        if num_selected > 0:
            # Для каждого выбранного токена решаем, что с ним делать
            action_prob = torch.rand(num_selected)  # [0,1)

            # Маски внутри выбранных позиций для разных действий
            mask_action = action_prob < 0.8
            random_action = (action_prob >= 0.8) & (action_prob < 0.9)
            keep_action = action_prob >= 0.9

            # Получаем индексы выбранных токенов
            selected_indices = selected.nonzero(as_tuple=True)[0]

            # Действие 1: замена на [MASK] (80%)
            mask_indices = selected_indices[mask_action]
            src_seq[mask_indices] = self.mask_token_id

            # Действие 2: замена на случайный токен (10%)
            random_indices = selected_indices[random_action]
            if random_indices.numel() > 0:
                # Случайный токен из диапазона байтов 0-255 (или всего словаря)
                random_tokens = torch.randint(5, 262, (random_indices.numel(),),
                                              dtype=torch.long)
                src_seq[random_indices] = random_tokens

            # Действие 3: оставить как есть (10%) – src_seq уже содержит оригинал

            # Заполняем target: только для выбранных позиций сохраняем оригинал
            tgt_seq[selected] = original[selected]

        return tgt_seq, src_seq


class MultiFileDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        mask_prob: float = 0.2,          # доля предсказываемых токенов
        extensions: tuple = ('.txt', '.enwik8', '.text', '.i', '.j'),
        cache_dir: str = None,
        force_rebuild: bool = False,
    ):
        self.tokenizer = ByteLevelTokenizer()
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.mask_token_id = self.tokenizer.encode("<MASK>")[0]

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {data_dir}")

        # Собираем все файлы с заданными расширениями
        self.files = sorted([
            p for p in self.data_dir.glob('**/*')
            if p.is_file() and (not extensions or p.suffix.lower() in extensions)
        ])

        if not self.files:
            raise ValueError(f"В {data_dir} нет файлов с расширениями {extensions}")

        # Генерируем уникальный хэш на основе путей и размеров файлов + параметров
        hash_str = self._compute_hash(seq_len, mask_prob)
        hash_id = hashlib.md5(hash_str.encode()).hexdigest()

        if cache_dir is None:
            cache_dir = self.data_dir
        else:
            cache_dir = Path(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        self.cache_path = cache_dir / f"multifile_{hash_id}.safetensors"

        if not force_rebuild and self.cache_path.exists():
            self.safetensors = safe_open(str(self.cache_path), framework="pt", device="cpu")
            self.num_samples = len([k for k in self.safetensors.keys() if k.startswith("input_ids_")])
        else:
            self._build_cache()

    def _compute_hash(self, seq_len: int, mask_prob: float) -> str:
        """Формирует строку для хэширования на основе файлов и параметров."""
        parts = [str(seq_len), str(mask_prob)]
        for f in self.files:
            stat = f.stat()
            parts.append(f"{f.resolve()}_{stat.st_size}_{stat.st_mtime}")
        return "|".join(parts)

    def _build_cache(self):
        full_tokens = []
        for file_path in self.files:
            try:
                with open(file_path, 'rb') as f:
                    byte_data = f.read()
                hex_str = byte_data.hex()
                tokens = self.tokenizer.encode(hex_str)
                full_tokens.extend(tokens)
            except Exception as e:
                print(f"Ошибка при чтении {file_path}: {e}. Пропускаем.")

        total_len = len(full_tokens)
        samples = []
        start = 0

        # Разбиваем на блоки по seq_len без перекрытия
        while start + self.seq_len <= total_len:
            chunk = full_tokens[start:start + self.seq_len]
            samples.append(chunk)
            start += self.seq_len
        # Хвост короче seq_len игнорируем

        # Сохраняем в safetensors
        tensor_dict = {}
        for i, seq in enumerate(samples):
            tensor_dict[f"input_ids_{i}"] = torch.tensor(seq, dtype=torch.long)

        self.num_samples = len(samples)
        save_file(tensor_dict, str(self.cache_path))
        self.safetensors = safe_open(str(self.cache_path), framework="pt", device="cpu")

        del full_tokens, samples, tensor_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Исходная последовательность
        original = self.safetensors.get_tensor(f"input_ids_{idx}").clone()
        src_seq = original.clone()
        tgt_seq = torch.full_like(original, -100)  # по умолчанию игнорируем всё

        # Выбираем токены для предсказания (mask_prob)
        prob = torch.rand(original.shape)
        selected = prob < self.mask_prob          # маска предсказываемых позиций
        num_selected = selected.sum().item()

        if num_selected > 0:
            # Для каждого выбранного токена решаем, что с ним делать
            action_prob = torch.rand(num_selected)

            mask_action = action_prob < 0.8
            random_action = (action_prob >= 0.8) & (action_prob < 0.9)
            keep_action = action_prob >= 0.9

            selected_indices = selected.nonzero(as_tuple=True)[0]

            # Замена на [MASK] (80%)
            mask_indices = selected_indices[mask_action]
            src_seq[mask_indices] = self.mask_token_id

            # Замена на случайный токен (10%)
            random_indices = selected_indices[random_action]
            if random_indices.numel() > 0:
                random_tokens = torch.randint(5, 262, (random_indices.numel(),),
                                              dtype=torch.long)
                src_seq[random_indices] = random_tokens

            # Оставить как есть (10%) – src_seq уже содержит оригинал

            # Target: только для выбранных позиций сохраняем оригинал
            tgt_seq[selected] = original[selected]

        return tgt_seq, src_seq
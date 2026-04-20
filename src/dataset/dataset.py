import os
import torch
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset
from safetensors.torch import save_file, load_file
from safetensors import safe_open
import hashlib
from model import ByteLevelTokenizer


def compute_entropy(tokens):
    """Вычисляет энтропию распределения токенов (в битах)."""
    counter = Counter(tokens)
    total = len(tokens)
    probs = [count / total for count in counter.values()]
    entropy = -sum(p * torch.log2(torch.tensor(p)) for p in probs)
    return entropy.item()


class FileDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: int,
        mask_prob: float = 0.2,
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

        params = (
            f"{os.path.basename(file_path)}_{seq_len}_"
            f"{mask_prob}_{os.path.getsize(file_path)}"
        )
        hash_id = hashlib.md5(params.encode()).hexdigest()
        self.cache_path = os.path.join(cache_dir, f"hexds_{hash_id}.safetensors")

        if not force_rebuild and os.path.exists(self.cache_path):
            self.safetensors = safe_open(self.cache_path, framework="pt", device="cpu")
            self.num_samples = len([k for k in self.safetensors.keys() if k.startswith("input_ids_")])
            # Чтение сохранённой энтропии
            try:
                entropy = self.safetensors.get_tensor("entropy").item()
                print(f"Loaded dataset from cache. Entropy: {entropy:.4f} bits")
            except Exception:
                print("Entropy not found in cache. Computing from loaded data (may be slow)...")
                # Вычисляем по загруженным тензорам
                all_tokens = []
                for i in range(self.num_samples):
                    all_tokens.extend(self.safetensors.get_tensor(f"input_ids_{i}").tolist())
                entropy = compute_entropy(all_tokens)
                print(f"Computed entropy: {entropy:.4f} bits")
        else:
            self._build_cache(file_path)

    def _build_cache(self, file_path):
        with open(file_path, 'rb') as f:
            byte_data = f.read()
        hex_str = byte_data.hex()
        full_tokens = self.tokenizer.encode(hex_str)

        # Вычисляем энтропию до разбиения на сэмплы
        entropy = compute_entropy(full_tokens)
        print(f"Dataset entropy: {entropy:.4f} bits")

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
        tensor_dict["entropy"] = torch.tensor(entropy)

        self.num_samples = len(samples)
        save_file(tensor_dict, self.cache_path)
        self.safetensors = safe_open(self.cache_path, framework="pt", device="cpu")

        del full_tokens, samples, tensor_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        original = self.safetensors.get_tensor(f"input_ids_{idx}").clone()
        src_seq = original.clone()
        tgt_seq = torch.full_like(original, -100)

        prob = torch.rand(original.shape)
        selected = prob < self.mask_prob
        num_selected = selected.sum().item()

        if num_selected > 0:
            action_prob = torch.rand(num_selected)
            mask_action = action_prob < 0.8
            random_action = (action_prob >= 0.8) & (action_prob < 0.9)
            keep_action = action_prob >= 0.9

            selected_indices = selected.nonzero(as_tuple=True)[0]

            mask_indices = selected_indices[mask_action]
            src_seq[mask_indices] = self.mask_token_id

            random_indices = selected_indices[random_action]
            if random_indices.numel() > 0:
                random_tokens = torch.randint(0, 256, (random_indices.numel(),),
                                              dtype=torch.long)
                src_seq[random_indices] = random_tokens

            tgt_seq[selected] = original[selected]

        return tgt_seq, src_seq


class MultiFileDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        mask_prob: float = 0.2,
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

        self.files = sorted([
            p for p in self.data_dir.glob('**/*')
            if p.is_file() and (not extensions or p.suffix.lower() in extensions)
        ])

        if not self.files:
            raise ValueError(f"В {data_dir} нет файлов с расширениями {extensions}")

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
            try:
                entropy = self.safetensors.get_tensor("entropy").item()
                print(f"Loaded dataset from cache. Entropy: {entropy:.4f} bits")
            except Exception:
                print("Entropy not found in cache. Computing from loaded data...")
                all_tokens = []
                for i in range(self.num_samples):
                    all_tokens.extend(self.safetensors.get_tensor(f"input_ids_{i}").tolist())
                entropy = compute_entropy(all_tokens)
                print(f"Computed entropy: {entropy:.4f} bits")
        else:
            self._build_cache()

    def _compute_hash(self, seq_len: int, mask_prob: float) -> str:
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

        entropy = compute_entropy(full_tokens)
        print(f"Dataset entropy: {entropy:.4f} bits")

        total_len = len(full_tokens)
        samples = []
        start = 0

        while start + self.seq_len <= total_len:
            chunk = full_tokens[start:start + self.seq_len]
            samples.append(chunk)
            start += self.seq_len

        tensor_dict = {}
        for i, seq in enumerate(samples):
            tensor_dict[f"input_ids_{i}"] = torch.tensor(seq, dtype=torch.long)
        tensor_dict["entropy"] = torch.tensor(entropy)

        self.num_samples = len(samples)
        save_file(tensor_dict, str(self.cache_path))
        self.safetensors = safe_open(str(self.cache_path), framework="pt", device="cpu")

        del full_tokens, samples, tensor_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        original = self.safetensors.get_tensor(f"input_ids_{idx}").clone()
        src_seq = original.clone()
        tgt_seq = torch.full_like(original, -100)

        prob = torch.rand(original.shape)
        selected = prob < self.mask_prob
        num_selected = selected.sum().item()

        if num_selected > 0:
            action_prob = torch.rand(num_selected)
            mask_action = action_prob < 0.8
            random_action = (action_prob >= 0.8) & (action_prob < 0.9)
            keep_action = action_prob >= 0.9

            selected_indices = selected.nonzero(as_tuple=True)[0]

            mask_indices = selected_indices[mask_action]
            src_seq[mask_indices] = self.mask_token_id

            random_indices = selected_indices[random_action]
            if random_indices.numel() > 0:
                random_tokens = torch.randint(0, 256, (random_indices.numel(),),
                                              dtype=torch.long)
                src_seq[random_indices] = random_tokens

            tgt_seq[selected] = original[selected]

        return tgt_seq, src_seq
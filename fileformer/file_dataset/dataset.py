import os
import torch
from torch.utils.data import Dataset
from safetensors.torch import save_file
from safetensors import safe_open
import hashlib
from fileformer.tokenizer import ByteLevelTokenizer

class FileDataset(Dataset):
    def __init__(self, file_path: str, cache_dir: str = None):
        pass

class ENWIK8Dataset(Dataset):
    def __init__(self, file_path: str, seq_len: int, overlap: int, cache_dir=None, force_rebuild=False):
        self.tokenizer = ByteLevelTokenizer()
        self.seq_len = seq_len
        self.overlap = overlap
        self.stride = seq_len - overlap

        if self.stride <= 0:
            raise ValueError("overlap должно быть меньше seq_len")

        if cache_dir is None:
            cache_dir = os.path.dirname(file_path)
        os.makedirs(cache_dir, exist_ok=True)

        params = f"{os.path.basename(file_path)}_{seq_len}_{overlap}_{os.path.getsize(file_path)}"
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
        masks = []
        total_len = len(full_tokens)
        start = 0

        while start + self.seq_len <= total_len:
            chunk = full_tokens[start:start + self.seq_len]
            samples.append(chunk)
            masks.append(torch.zeros(self.seq_len, dtype=torch.long))
            start += self.stride

        if start < total_len:
            chunk = full_tokens[start:]
            pad_len = self.seq_len - len(chunk)
            chunk += [self.tokenizer.encode("<pad>")[0]] * pad_len
            last_mask = torch.zeros(self.seq_len, dtype=torch.long)
            last_mask[-pad_len:] = 1
            masks.append(last_mask)
            samples.append(chunk)

        tensor_dict = {}
        for i, seq in enumerate(samples):
            tensor_dict[f"input_ids_{i}"] = torch.tensor(seq, dtype=torch.long)
            tensor_dict[f"attention_mask_{i}"] = masks[i]

        self.num_samples = len(samples)
        save_file(tensor_dict, self.cache_path)
        self.safetensors = safe_open(self.cache_path, framework="pt", device="cpu")

        del full_tokens, samples, tensor_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = self.safetensors.get_tensor(f"input_ids_{idx}")
        attention_mask = self.safetensors.get_tensor(f"attention_mask_{idx}")
        causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        return input_ids, attention_mask, causal_mask
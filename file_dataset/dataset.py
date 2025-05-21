import zlib
import torch
from tqdm import tqdm
from pathlib import Path
from hashlib import sha256, md5
from tokenizer import Tokenizer
from torch.utils.data import Dataset

class FileDataset(Dataset):
    def __init__(self, path, hash_type, max_bytes):
        self.tokenizer = Tokenizer()
        self.files = [x for x in Path(path).glob('**/*') if x.is_file()]
        self.max_bytes = max_bytes

        # Выбор хеш-функции
        if hash_type == 'md5':
            self.hash = md5
        elif hash_type == 'sha256':
            self.hash = sha256
        elif hash_type == 'crc32':
            self.hash = lambda x: zlib.crc32(x) & 0xFFFFFFFF
        else:
            raise ValueError("Unsupported hash type")

        self.data = self.prepare(self.files, self.max_bytes, self.hash)

    def prepare(self, files, max_bytes, hash_fn):
        data = []
        for file in tqdm(files):
            with open(file, 'rb') as f:
                data_from_file = f.read()
                chunks = [
                    data_from_file[i:i + max_bytes]
                    for i in range(0, len(data_from_file), max_bytes)
                ]
            for chunk in chunks:
                entry = {
                    'hash': hash_fn(chunk).hexdigest() if hasattr(hash_fn, 'hexdigest') else f"{hash_fn(chunk):08x}",
                    'hex_data': chunk.hex(),
                }
                # Токенизация и паддинг
                tokens = self.tokenizer.encode(chunk)
                if len(tokens) < (max_bytes // 2):  # 2 байта на токен
                    pad_size = (max_bytes // 2) - len(tokens)
                    tokens += [self.tokenizer.get_idx_from_token('<PAD>')] * pad_size
                entry['tokens'] = tokens
                data.append(entry)
        return data

#    def __len__(self):
#        return len(self.data)
#
#    def __getitem__(self, idx):
#        item = self.data[idx]
#        return torch.tensor(item['tokens']), item['hash']
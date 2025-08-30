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

class FileDataset(Dataset):
    def __init__(self, path, seq_len, batch_size):
        self.files = [x for x in Path(path).glob('**/*') if x.is_file()]
        self.tokenizer = ByteLevelTokenizer()
        self.seq_len = seq_len
        self.pad_token = self.tokenizer.encode("<pad>")[0]
        self.mask_token = self.tokenizer.encode("<mask>")[0]
        self.table_init_data = pd.DataFrame(columns=["id", "file_name", "start_byte", "end_byte"], index=[0])

    def prepare(self):
        byte_len = 0
        for i in range(len(self.files)):
            num_chuncs_of_seq_len = (self.files[i].stat().st_size//self.seq_len)
            new_data = pd.DataFrame({'id': i, 'file_name': str(self.files[i]), 'start_byte': byte_len, 'end_byte':num_chuncs_of_seq_len+byte_len}, index=[0])
            self.table_init_data = pd.concat([self.table_init_data, new_data])
            byte_len = num_chuncs_of_seq_len+byte_len+1

    def get_file_name_by_byte(self, byte_value):
        for index, row in self.table_init_data.iterrows():
            if row['start_byte'] <= byte_value <= row['end_byte']:
                return row['file_name'], row['start_byte'], row['end_byte']
        return None

    def read_file(self, file_name, start_pos):
        pads = torch.ones((1, self.seq_len))
        with open(file_name, 'rb') as f:
            f.seek(start_pos)
            data = f.read(self.seq_len).hex()
            tokens = self.tokenizer.encode(data)
            if len(tokens) < self.seq_len:
                pads = torch.tensor([[1]*len(tokens), [0]*(self.seq_len - len(tokens))])
                tokens.extend([self.pad_token]*(self.seq_len - len(tokens)))
        f.close()
        return torch.tensor(tokens, dtype=torch.long), pads

    def mask_tokens(self, x):
        masked = x.clone()
        for i in range(self.seq_len):
            if masked[i] == self.pad_token:
                pass
            if random.random() < 0.5:
                masked[i] = torch.tensor(self.mask_token)
        return masked

    def get_hesh_tokens(self, file_name):
        sha256_hash = hashlib.new('sha256')
        with open(file_name, 'rb') as f:
            while True:
                data = f.read(1024)
                if not data:
                    break
                sha256_hash.update(data)
        sha256_hash = sha256_hash.hexdigest()
        sha256_hash_tokens = self.tokenizer.encode(sha256_hash)
        return torch.tensor(sha256_hash_tokens, dtype=torch.long)

    def __len__(self):
        return self.table_init_data['end_byte'].max()

    def __getitem__(self, index):
        info = self.get_file_name_by_byte(index)
        tokens, pads = self.read_file(info[0], info[1])
        masked_tokens = self.mask_tokens(tokens)
        hash = self.get_hesh_tokens(info[0])

        extention_tokenize = torch.tensor([get_file_ext_as_token(info[0])], dtype=torch.long)

        return tokens, masked_tokens, pads, hash, extention_tokenize

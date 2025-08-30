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
        # Initialize dataset with file path, sequence length, and batch size
        self.files = [x for x in Path(path).glob('**/*') if x.is_file()]
        self.tokenizer = ByteLevelTokenizer()  # Initialize byte-level tokenizer
        self.seq_len = seq_len  # Set sequence length for tokenization
        self.pad_token = self.tokenizer.encode("<pad>")[0]  # Token for padding
        self.mask_token = self.tokenizer.encode("<mask>")[0]  # Token for masking
        self.table_init_data = pd.DataFrame(columns=["id", "file_name", "start_byte", "end_byte"], index=[0])  # DataFrame to track file metadata

    def prepare(self):
        # Prepare metadata for files, mapping byte ranges to files
        byte_len = 0
        for i in range(len(self.files)):
            num_chuncs_of_seq_len = (self.files[i].stat().st_size // self.seq_len)  # Calculate number of sequence-length chunks
            new_data = pd.DataFrame({
                'id': i,
                'file_name': str(self.files[i]),
                'start_byte': byte_len,
                'end_byte': num_chuncs_of_seq_len + byte_len
            }, index=[0])
            self.table_init_data = pd.concat([self.table_init_data, new_data])  # Append file metadata
            byte_len = num_chuncs_of_seq_len + byte_len + 1

    def get_file_name_by_byte(self, byte_value):
        # Retrieve file name and byte range for a given byte value
        for index, row in self.table_init_data.iterrows():
            if row['start_byte'] <= byte_value <= row['end_byte']:
                return row['file_name'], row['start_byte'], row['end_byte']
        return None  # Return None if no matching file is found

    def read_file(self, file_name, start_pos):
        # Read and tokenize a chunk of a file starting at start_pos
        pads = torch.ones((1, self.seq_len))  # Initialize padding tensor
        with open(file_name, 'rb') as f:
            f.seek(start_pos)  # Move to the specified byte position
            data = f.read(self.seq_len).hex()  # Read and convert to hex
            tokens = self.tokenizer.encode(data)  # Tokenize the data
            if len(tokens) < self.seq_len:
                pads = torch.tensor([[1] * len(tokens) + [0] * (self.seq_len - len(tokens))])  # Create padding mask
                tokens.extend([self.pad_token] * (self.seq_len - len(tokens)))  # Pad tokens to seq_len
        return torch.tensor(tokens, dtype=torch.long), pads

    def mask_tokens(self, x):
        # Randomly mask tokens (except padding tokens) with a 50% probability
        masked = x.clone()
        for i in range(self.seq_len):
            if masked[i] == self.pad_token:
                continue  # Skip padding tokens
            if random.random() < 0.5:
                masked[i] = torch.tensor(self.mask_token)  # Replace with mask token
        return masked

    def get_hesh_tokens(self, file_name):
        # Compute SHA-256 hash of a file and tokenize it
        sha256_hash = hashlib.new('sha256')
        with open(file_name, 'rb') as f:
            while True:
                data = f.read(1024)  # Read file in 1KB chunks
                if not data:
                    break
                sha256_hash.update(data)
        sha256_hash = sha256_hash.hexdigest()  # Get hex representation of hash
        sha256_hash_tokens = self.tokenizer.encode(sha256_hash)  # Tokenize the hash
        return torch.tensor(sha256_hash_tokens, dtype=torch.long)

    def __len__(self):
        # Return the total number of byte chunks across all files
        return self.table_init_data['end_byte'].max()

    def __getitem__(self, index):
        # Retrieve a dataset item for a given index
        info = self.get_file_name_by_byte(index)  # Get file info for the index
        tokens, pads = self.read_file(info[0], info[1])  # Read and tokenize file chunk
        masked_tokens = self.mask_tokens(tokens)  # Apply random masking
        hash = self.get_hesh_tokens(info[0])  # Get tokenized file hash
        extention_tokenize = torch.tensor([get_file_ext_as_token(info[0])], dtype=torch.long)  # Tokenize file extension
        return tokens, masked_tokens, pads, hash, extention_tokenize  # Return dataset item
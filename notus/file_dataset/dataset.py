import torch
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

class FileDataset(Dataset):
    def __init__(self, path, max_seq_length, tokenizer):
        self.tokenizer = tokenizer
        self.files = [x for x in Path(path).glob('**/*') if x.is_file()]
        self.max_seq_length = max_seq_length
        self.pad_token = self.tokenizer.encode("<pad>")[0]
        self.mask_token = self.tokenizer.encode("<mask>")[0]
        self.file_token_lengths = []
        self.total_chunks = 0
        for file in tqdm(self.files):
            all_tokens = self.read_and_tokenize(file)
            num_tokens = len(all_tokens)
            num_chunks = (num_tokens + self.max_seq_length - 1) // self.max_seq_length
            self.file_token_lengths.append(num_tokens)
            self.total_chunks += num_chunks

    def read_and_tokenize(self, file):
        all_tokens = []
        with open(file, 'rb') as f:
            while True:
                byte_chunk = f.read(4096)
                if not byte_chunk:
                    break
                hex_chunk = byte_chunk.hex()
                tokens = self.tokenizer.encode(hex_chunk)
                all_tokens.extend(tokens)
        return all_tokens

    def transform_tensor(self, input_tensor):
        """Улучшенная функция маскирования с возвратом лейблов"""
        seq_len = input_tensor.size(0)
        output_tensor = input_tensor.clone()
        labels = torch.full_like(input_tensor, fill_value=-100)

        i = 0
        while i < seq_len:
            k = random.choice([0, 1, 2, 4])
            start_index = i + 1
            end_index = min(start_index + k, seq_len)

            for j in range(start_index, end_index):
                labels[j] = input_tensor[j].item()
                output_tensor[j] = self.mask_token

            i = end_index

        return output_tensor, labels

    def pad_tensors(self, tensor, labels):
        real_len = len(tensor)
        real_pad = self.max_seq_length - real_len
        padded_tensor = F.pad(tensor, (0, real_pad), value=self.pad_token)
        padded_labels = F.pad(labels, (0, real_pad), value=-100)
        attention_mask = torch.ones(real_len, dtype=torch.int8)
        attention_mask = F.pad(attention_mask, (0, real_pad), value=0)
        return padded_tensor, padded_labels, attention_mask

    def pad_sequence(self, tensor):
        real_len = len(tensor)
        real_pad = self.max_seq_length - real_len
        padded_tensor = F.pad(tensor, (0, real_pad), value=self.pad_token)
        attention_mask = torch.ones(real_len, dtype=torch.int8)
        attention_mask = F.pad(attention_mask, (0, real_pad), value=0)
        return padded_tensor, attention_mask

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        if idx >= self.total_chunks or idx < 0:
            raise IndexError(f"Index {idx} out of range")

        cum_chunks = 0
        for f_idx, num_tokens in enumerate(self.file_token_lengths):
            num_chunks = (num_tokens + self.max_seq_length - 1) // self.max_seq_length
            if idx < cum_chunks + num_chunks:
                local_chunk_idx = idx - cum_chunks
                break
            cum_chunks += num_chunks

        all_tokens = self.read_and_tokenize(self.files[f_idx])

        start = local_chunk_idx * self.max_seq_length
        end = min(start + self.max_seq_length, len(all_tokens))
        real_chunk = all_tokens[start:end]

        real_tensor = torch.tensor(real_chunk, dtype=torch.int16)
        ori_padded, attn_mask_ori = self.pad_sequence(real_tensor)

        masked_tensor, labels = self.transform_tensor(real_tensor.clone())
        masked_padded, labels_padded, attn_mask = self.pad_tensors(masked_tensor, labels)

        mask_masked = (labels_padded != -100).to(torch.int8)

        return {
            'lbl': labels_padded,
            'iid': masked_padded,
            'ori': ori_padded,
            'attn_mask': attn_mask,
            'attn_mask_ori': attn_mask_ori,
            'mask': mask_masked
        }
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
        self.data = self.prepare(self.files)

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

    def prepare(self, files):
        all_chunks = []
        for file in tqdm(files):
            try:
                all_tokens = self.read_and_tokenize(file)
                chunk_data = self.process_chunks(all_tokens)
                all_chunks.extend(chunk_data)
            except Exception as e:
                print(f"Ошибка обработки {file}: {e}")
                continue
        return all_chunks

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

    def process_chunks(self, all_tokens):
        chunk_data = []
        for i in range(0, len(all_tokens), self.max_seq_length):
            real_chunk = all_tokens[i:i + self.max_seq_length]
            real_tensor = torch.tensor(real_chunk, dtype=torch.int16)
            real_tensor, real_tensor_mask = self.pad_sequence(real_tensor)
            masked_tensor, labels = self.transform_tensor(real_tensor.clone())
            padded_masked, padded_labels, attention_mask = self.pad_tensors(masked_tensor, labels)
            chunk_data.append({
                'lbl': padded_labels,
                'iid': padded_masked,
                'ori': real_tensor,
                'attn_mask': attention_mask,
                'attn_mask_ori': real_tensor_mask
            })
        return chunk_data

    def pad_tensors(self, tensor, labels):
        real_len = len(tensor)
        real_pad = self.max_seq_length - real_len
        padded_tensor = F.pad(tensor, (0, real_pad), value=self.pad_token)
        padded_labels = F.pad(labels, (0, real_pad), value=-100)
        # Create attention mask: 1 for real tokens, 0 for padding
        attention_mask = torch.ones(real_len, dtype=torch.int8)
        attention_mask = F.pad(attention_mask, (0, real_pad), value=0)
        return padded_tensor, padded_labels, attention_mask

    def pad_sequence(self, tensor):
        """
        Adds padding to a tensor of sequences to match the maximum sequence length.

        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, seq_len].
            max_seq_len (int): Maximum sequence length to pad to.
            padding_value (int, optional): Value to use for padding. Defaults to 0.

        Returns:
            torch.Tensor: Padded tensor of shape [batch_size, max_seq_len].
            torch.Tensor: Padding mask of shape [batch_size, max_seq_len], where True indicates padding.
        """
        batch_size, seq_len = tensor.shape

        if seq_len > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_length}")

        if seq_len == self.max_seq_length:
            padding_mask = torch.zeros_like(tensor, dtype=torch.bool)
            return tensor, padding_mask

        # Create padded tensor
        padded_tensor = torch.full(
            (batch_size, self.max_seq_length),
            fill_value=self.pad_token,
            dtype=tensor.dtype,
            device=tensor.device
        )
        padded_tensor[:, :seq_len] = tensor

        # Create padding mask (True for padding positions)
        padding_mask = torch.ones(
            (batch_size, self.max_seq_length),
            dtype=torch.bool,
            device=tensor.device
        )
        padding_mask[:, :seq_len] = False

        return padded_tensor, padding_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
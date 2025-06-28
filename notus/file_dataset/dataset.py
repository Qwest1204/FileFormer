import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
import random
import torch.nn.functional as F  # Добавляем импорт для F.pad

class FileDataset(Dataset):
    def __init__(self, path, max_seq_length, tokenizer):
        self.tokenizer = tokenizer
        self.files = [x for x in Path(path).glob('**/*') if x.is_file()]
        self.max_seq_length = max_seq_length
        self.pad_token = self.tokenizer.get_special_token_id("<pad>")
        self.mask_token = self.tokenizer.get_special_token_id("<mask>")
        self.data = self.prepare(self.files)

    @staticmethod
    def transform_tensor(input_tensor, value):
        if value is None:
            replace_value = torch.tensor(0, dtype=input_tensor.dtype)
        elif not isinstance(value, torch.Tensor):
            replace_value = torch.tensor(value, dtype=input_tensor.dtype)
        else:
            replace_value = value

        output_tensor = input_tensor.clone()
        seq_len = input_tensor.size(0)
        replaced_elements = []

        i = 0
        while i < seq_len:
            k = random.choice([0, 1, 2, 4, 8, 16])
            start_index = i + 1
            end_index = min(start_index + k, seq_len)

            for j in range(start_index, end_index):
                replaced_elements.append(input_tensor[j].item())
                output_tensor[j] = replace_value

            i = end_index

        if replaced_elements:
            replaced_tensor = torch.tensor(replaced_elements, dtype=input_tensor.dtype)
        else:
            replaced_tensor = torch.tensor([], dtype=input_tensor.dtype)

        return output_tensor, replaced_tensor

    def prepare(self, files):
        data = []
        for file in tqdm(files):
            with open(file, 'rb') as f:
                data_from_file = f.read().hex()
                tokens = self.tokenizer.encode(data_from_file)

                for i in range(0, len(tokens), self.max_seq_length):
                    real_data_chunk = torch.tensor(tokens[i:i + self.max_seq_length])
                    mask_data_chunk, removed_elements = self.transform_tensor(real_data_chunk, self.mask_token)

                    mask_data_chunk_len = len(mask_data_chunk)
                    removed_elements_len = len(removed_elements)
                    real_data_chunk_len = len(real_data_chunk)

                    pad_size1 = self.max_seq_length - mask_data_chunk_len
                    pad_size2 = self.max_seq_length - removed_elements_len
                    pad_size3 = self.max_seq_length - real_data_chunk_len

                    # Используем F.pad вместо torch.cat для безопасной работы с пустыми тензорами
                    if pad_size1 > 0:
                        mask_data_chunk = F.pad(mask_data_chunk, (0, pad_size1), value=self.pad_token)
                    if pad_size2 > 0:
                        removed_elements = F.pad(removed_elements, (0, pad_size2), value=self.pad_token)
                    if pad_size3 > 0:
                        real_data_chunk = F.pad(real_data_chunk, (0, pad_size3), value=self.pad_token)

                    attention_mask_removed_elements = [1] * removed_elements_len + [0] * pad_size2
                    attention_mask_mask_data_chunk = [1] * mask_data_chunk_len + [0] * pad_size1

                    data.append({
                        'origin_ids': real_data_chunk,
                        'masked_ids': mask_data_chunk,
                        'attention_mask_masked_ids': torch.tensor(attention_mask_mask_data_chunk),
                        'removed_elements': removed_elements,
                        'attention_mask_removed_elements': torch.tensor(attention_mask_removed_elements),
                    })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
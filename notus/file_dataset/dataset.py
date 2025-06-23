import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class FileDataset(Dataset):
    def __init__(self, path, max_seq_length, tokenizer):
        self.tokenizer = tokenizer
        self.files = [x for x in Path(path).glob('**/*') if x.is_file()]
        self.max_seq_length = max_seq_length  # В токенах (учитывая SOF/EOF)
        self.pad_token = self.tokenizer.get_special_token_id("<|PAD|>")
        self.mask_token = self.tokenizer.get_special_token_id("<|MASK|>")
        self.data = self.prepare(self.files)

    @staticmethod
    def transform_tensor(input_tensor, value):
        output_tensor = input_tensor.clone()
        seq_len = input_tensor.size(0)
        replaced_elements = []

        i = 0
        while i < seq_len:
            if i + 1 < seq_len:
                replaced_elements.append(input_tensor[i + 1].item())
                output_tensor[i + 1] = value
            if i + 2 < seq_len:
                replaced_elements.append(input_tensor[i + 2].item())
                output_tensor[i + 2] = value
            i += 3

        replaced_tensor = torch.tensor(replaced_elements,
                                       dtype=input_tensor.dtype) if replaced_elements else torch.tensor([],
                                                                                                        dtype=input_tensor.dtype)
        return output_tensor, replaced_tensor

    def prepare(self, files):
        data = []
        for file in tqdm(files):
            with open(file, 'rb') as f:
                data_from_file = f.read().hex()
                tokens = self.tokenizer.encode(data_from_file)

                # Разделение на последовательности с учетом max_length
                for i in range(0, len(tokens), self.max_seq_length):
                    real_data_chunk = torch.tensor(tokens[i:i + self.max_seq_length])
                    mask_data_chunk, removed_elements = self.transform_tensor(real_data_chunk, self.mask_token)

                    mask_data_chunk_len = len(mask_data_chunk) #need 512
                    removed_elements_len = len(removed_elements) #heed 341
                    real_data_chunk_len = len(real_data_chunk) #need 512

                    pad_size1 = self.max_seq_length - mask_data_chunk_len
                    pad_size2 = self.max_seq_length - removed_elements_len
                    pad_size3 = self.max_seq_length - real_data_chunk_len

                    # Добавляем паддинг только если необходимо
                    if pad_size1 > 0:
                        mask_data_chunk = torch.cat((mask_data_chunk, torch.tensor([self.pad_token] * pad_size1)))
                    if pad_size2 > 0:
                        removed_elements = torch.cat((removed_elements, torch.tensor([self.pad_token] * pad_size2)))
                    if pad_size3 > 0:
                        real_data_chunk = torch.cat((real_data_chunk, torch.tensor([self.pad_token] * pad_size3)))

                    # Создание масок внимания (1 для реальных токенов, 0 для паддинга)
                    attention_mask_removed_elements = [1] * removed_elements_len + [0] * pad_size2
                    attention_mask_mask_data_chunk = [1] * mask_data_chunk_len + [0] * pad_size1

                    # Для трансформера: input_ids и labels (сдвинутые на 1 токен)
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

    #@staticmethod
    #def collate_fn(batch):
    #    return {
    #        'input_ids': pad_sequence([item['input_ids'] for item in batch], batch_first=True),
    #        'attention_mask': pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    #    }
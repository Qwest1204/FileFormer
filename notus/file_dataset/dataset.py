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
        self.data = self.prepare(self.files)

    def prepare(self, files):
        data = []
        for file in tqdm(files):
            with open(file, 'rb') as f:
                data_from_file = f.read().hex()
                tokens = self.tokenizer.encode(data_from_file)

                # Разделение на последовательности с учетом max_length
                for i in range(0, len(tokens), self.max_seq_length):
                    chunk = tokens[i:i + self.max_seq_length]
                    current_length = len(chunk)
                    pad_size = self.max_seq_length - current_length  # Всегда вычисляем pad_size

                    # Добавляем паддинг только если необходимо
                    if pad_size > 0:
                        chunk += [self.pad_token] * pad_size

                    # Создание масок внимания (1 для реальных токенов, 0 для паддинга)
                    attention_mask = [1] * current_length + [0] * pad_size

                    # Для трансформера: input_ids и labels (сдвинутые на 1 токен)
                    data.append({
                        'input_ids': torch.tensor(chunk),
                        'attention_mask': torch.tensor(attention_mask),
                    })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(batch):
        return {
            'input_ids': pad_sequence([item['input_ids'] for item in batch], batch_first=True),
            'attention_mask': pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
        }
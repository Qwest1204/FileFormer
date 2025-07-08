import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
import numpy as np


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
        labels = torch.full_like(input_tensor, fill_value=-100)  # -100 для игнорирования в loss

        i = 0
        while i < seq_len:
            k = random.choice([0, 1, 2, 4, 8, 16])
            start_index = i + 1
            end_index = min(start_index + k, seq_len)

            for j in range(start_index, end_index):
                # Сохраняем оригинальный токен для лейбла
                labels[j] = input_tensor[j].item()
                # Заменяем на маскированный токен
                output_tensor[j] = self.mask_token

            i = end_index

        return output_tensor, labels

    def prepare(self, files):
        full_files = []
        for file in tqdm(files):
            ff_data = []
            ff_attmask = []
            chunk_data = []

            try:
                # Чтение и токенизация файла
                all_tokens = []
                with open(file, 'rb') as f:
                    while True:
                        byte_chunk = f.read(4096)
                        if not byte_chunk:
                            break
                        hex_chunk = byte_chunk.hex()
                        tokens = self.tokenizer.encode(hex_chunk)
                        all_tokens.extend(tokens)

                # Обработка чанков
                for i in range(0, len(all_tokens), self.max_seq_length):
                    real_chunk = all_tokens[i:i + self.max_seq_length]
                    real_tensor = torch.tensor(real_chunk, dtype=torch.long)
                    real_len = len(real_tensor)

                    # Маскирование с получением лейблов
                    masked_tensor, labels = self.transform_tensor(real_tensor.clone())

                    # Паддинг
                    real_pad = self.max_seq_length - real_len
                    padded_real = F.pad(real_tensor, (0, real_pad), value=self.pad_token)
                    padded_masked = F.pad(masked_tensor, (0, real_pad), value=self.pad_token)
                    padded_labels = F.pad(labels, (0, real_pad), value=-100)  # Паддинг игнорируется

                    # Маски внимания
                    real_attmask = torch.cat([
                        torch.ones(real_len, dtype=torch.float32),
                        torch.zeros(real_pad, dtype=torch.float32)
                    ])

                    ff_data.append(padded_real)
                    ff_attmask.append(real_attmask)

                    chunk_data.append({
                        'input_ids': padded_masked,
                        'attention_mask': real_attmask.clone(),
                        'labels': padded_labels
                    })

            except Exception as e:
                print(f"Ошибка обработки {file}: {e}")
                continue

            if ff_data:
                full_files.append({
                    'full_file_tokens': torch.stack(ff_data),
                    'full_file_attmask': torch.stack(ff_attmask),
                    'chunks': chunk_data
                })
            else:
                print(f"Файл {file} не содержит данных")

        return full_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(batch):
        """Функция для объединения данных в батчи"""
        full_file_tokens = [item['full_file_tokens'] for item in batch]
        full_file_attmask = [item['full_file_attmask'] for item in batch]

        # Для чанков создаем "мега-батч" из всех чанков
        chunks_data = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for item in batch:
            for chunk in item['chunks']:
                chunks_data['input_ids'].append(chunk['input_ids'])
                chunks_data['attention_mask'].append(chunk['attention_mask'])
                chunks_data['labels'].append(chunk['labels'])

        return {
            'full_file_tokens': full_file_tokens,
            'full_file_attmask': full_file_attmask,
            'chunks': {
                'input_ids': torch.stack(chunks_data['input_ids']),
                'attention_mask': torch.stack(chunks_data['attention_mask']),
                'labels': torch.stack(chunks_data['labels'])
            }
        }
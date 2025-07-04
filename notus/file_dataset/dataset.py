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
        self.pad_token = self.tokenizer.encode("<pad>")[0]
        self.mask_token = self.tokenizer.encode("<mask>")[0]
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

        return output_tensor

    def prepare(self, files):
        full_files = []
        for file in tqdm(files):
            # Основные контейнеры для файла
            ff_data = []  # Исходные токены (Nx512)
            ff_attmask = []  # Маски внимания для исходных токенов (Nx512)
            chunk_data = []  # Словари для каждого чанка

            try:
                # Чтение файла блоками для обработки больших файлов
                chunk_size = 4096  # 4KB блоки
                all_tokens = []
                with open(file, 'rb') as f:
                    while True:
                        byte_chunk = f.read(chunk_size)
                        if not byte_chunk:
                            break
                        hex_chunk = byte_chunk.hex()
                        tokens = self.tokenizer.encode(hex_chunk)
                        all_tokens.extend(tokens)

                # Обработка чанков
                for i in range(0, len(all_tokens), self.max_seq_length):
                    # Исходный чанк
                    real_chunk = all_tokens[i:i + self.max_seq_length]
                    real_tensor = torch.tensor(real_chunk, dtype=torch.long)
                    real_len = len(real_tensor)

                    # Маскированный чанк
                    masked_tensor = self.transform_tensor(real_tensor.clone(), self.mask_token)
                    if len(masked_tensor) > self.max_seq_length:
                        masked_tensor = masked_tensor[:self.max_seq_length]
                    masked_len = len(masked_tensor)

                    # Паддинг для исходных токенов
                    real_pad = self.max_seq_length - real_len
                    padded_real = F.pad(real_tensor, (0, real_pad), value=self.pad_token)

                    # Паддинг для маскированных токенов
                    masked_pad = self.max_seq_length - masked_len
                    padded_masked = F.pad(masked_tensor, (0, masked_pad), value=self.pad_token)

                    # Маски внимания
                    real_attmask = torch.cat([
                        torch.ones(real_len, dtype=torch.float32),
                        torch.zeros(real_pad, dtype=torch.float32)
                    ])

                    masked_attmask = torch.cat([
                        torch.ones(masked_len, dtype=torch.float32),
                        torch.zeros(masked_pad, dtype=torch.float32)
                    ])

                    # Сохранение данных
                    ff_data.append(padded_real)
                    ff_attmask.append(real_attmask)

                    chunk_data.append({
                        'chunk_masked': padded_masked,
                        'attention': masked_attmask,
                        'right_chunk': padded_real
                    })

            except Exception as e:
                print(f"Ошибка обработки {file}: {e}")
                continue

            # Сборка результата для файла
            if ff_data:
                file_dict = {
                    'fulfiletoken': torch.stack(ff_data),
                    'fullfileattmask': torch.stack(ff_attmask),
                    'data': chunk_data
                }
                full_files.append(file_dict)
            else:
                print(f"Файл {file} не содержит данных")

        return full_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
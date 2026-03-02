import os
import torch
import argparse

from pathlib import Path
from hashlib import sha256
from torch import Tensor
from safetensors.torch import save_file

from fileformer.tokenizer import ByteLevelTokenizer
from patterns import META_END_MARKERS

tokenizer = ByteLevelTokenizer()


parser = argparse.ArgumentParser(
                    prog='dataset preparation',
                    description='Optim to PNG/JPEG/MP4/OBJ/ZIP',
                    epilog='Optim to PNG/JPEG/MP4/OBJ/ZIP')

parser.add_argument('in_folder')
parser.add_argument('out_folder')
parser.add_argument('-b', '--bsize', default=2048, type=int)

args = parser.parse_args()

INPUT_ROOT = Path(args.in_folder)
OUTPUT_ROOT = Path(args.out_folder)
Path(OUTPUT_ROOT).mkdir(exist_ok=True)
CHUNK_SIZE = args.bsize

def _tensor_from_tokens(tokens, dtype=torch.float) -> Tensor:
    return torch.tensor(tokens, dtype=dtype)

def process_file(file_path: Path, output_base: Path):
    """Обрабатывает один файл: отделяет метаданные, сохраняет их и данные по частям."""
    # Создаём выходную директорию для файла
    out_dir = output_base / file_path.name
    out_dir.mkdir(exist_ok=True)

    with open(file_path, 'rb') as f:
        # --- Поиск маркера конца метаданных ---
        non_empty_markers = [m for m in META_END_MARKERS if m]
        meta_buffer = bytearray()
        data_remainder = b''

        if non_empty_markers:
            search_buf = bytearray()
            max_marker_len = max(len(m) for m in non_empty_markers)
            marker_found = False

            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                search_buf.extend(chunk)

                # Поиск первого вхождения любого маркера
                best_pos = None
                best_marker = None
                for marker in non_empty_markers:
                    pos = search_buf.find(marker)
                    if pos != -1:
                        if best_pos is None or pos < best_pos:
                            best_pos = pos
                            best_marker = marker

                if best_pos is not None:
                    # Маркер найден
                    meta_buffer.extend(search_buf[:best_pos])
                    data_remainder = search_buf[best_pos + len(best_marker):]
                    marker_found = True
                    break
                else:
                    # Сохраняем часть буфера, оставляя хвост для перекрытия
                    if len(search_buf) > max_marker_len:
                        save_len = len(search_buf) - max_marker_len
                        meta_buffer.extend(search_buf[:save_len])
                        # Оставляем только хвост, где может начаться маркер
                        search_buf = search_buf[save_len:]

            if not marker_found:
                # Маркер не найден – весь файл считаем метаданными
                meta_buffer.extend(search_buf)
                data_remainder = b''
        else:
            # Нет ни одного непустого маркера – метаданных нет, весь файл — данные
            meta_buffer = bytearray()
            # Файл ещё не читался, указатель в начале

        print(f"File: {file_path.name}, metadata size: {len(meta_buffer)} bytes")

        # Сохраняем метаданные
        if meta_buffer or True:  # всегда сохраняем, даже пустые (можно убрать условие)
            hash_hex = sha256(meta_buffer).hexdigest()
            hash_tokens = _tensor_from_tokens(tokenizer.encode(hash_hex))
            meta_tokens = _tensor_from_tokens(tokenizer.encode(meta_buffer.hex()))
            save_file(
                {'hash_tokens': hash_tokens, 'tokenized_metadata': meta_tokens},
                out_dir / 'meta.safetensors'
            )

        # --- Обработка данных чанками ---
        chunk_number = 0
        current_chunk = bytearray(data_remainder)  # остаток от буфера поиска

        while True:
            # Добираем данные до полного чанка (CHUNK_SIZE)
            while len(current_chunk) < CHUNK_SIZE:
                more = f.read(CHUNK_SIZE - len(current_chunk))
                if not more:
                    break
                current_chunk.extend(more)

            if current_chunk:
                # Вычисляем хеш и токенизируем данные чанка
                hash_hex = sha256(current_chunk).hexdigest()
                hash_tokens = _tensor_from_tokens(tokenizer.encode(hash_hex))
                data_tokens = _tensor_from_tokens(tokenizer.encode(current_chunk.hex()))

                save_file(
                    {'hash_tokens': hash_tokens, 'tokenized_data': data_tokens},
                    out_dir / f'data{chunk_number}.safetensors'
                )

                chunk_number += 1
                current_chunk = bytearray()  # готовим для следующего чанка
            else:
                break

        print(f"Total chunks: {chunk_number}\n")

for file in INPUT_ROOT.iterdir():
    if not file.is_file():
        continue
    process_file(file, OUTPUT_ROOT)
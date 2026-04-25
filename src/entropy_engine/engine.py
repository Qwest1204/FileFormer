#  ╔══════════════════════════════════════════════════════════════════╗
#  ║                  Purity Seal of the Omnissiah                    ║
#  ║  ❀  Let this code be sanctified, free from error and heresy.     ║
#  ║  ❀  By the Motive Force, let no null pointer or segfault arise.  ║
#  ║  ❀  The flesh is weak, but the logic endures.                    ║
#  ║  ❀  Praise the Machine God. May the binary be true.              ║
#  ║                                                                  ║
#  ║              (___)                                               ║
#  ║              (o o)   <  Ave Deus Mechanicus!                     ║
#  ║              /"\                                                 ║
#  ║             /_/ \_                                               ║
#  ║                                                                  ║
#  ║                     ~ Puritas Codicis ~                          ║
#  ╚══════════════════════════════════════════════════════════════════╝

import struct
import numpy as np
import torch
import constriction
from model import FileRWKV
from tokenizer import ByteLevelTokenizer
from utils.utils import normalize_probabilities

import tqdm


class Engine:
    """
    Compression / decompression engine using a FileRWKV model.
    Handles chunked byte‑level compression without a special start token.
    """

    def __init__(self, seed: int, nn_model: FileRWKV, tokenizer: ByteLevelTokenizer, chunk_size: int = 2048,
                 scale_factor: int = 24):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = nn_model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.model.eval()

        self.vocab_size = tokenizer.vocab_size
        # Fixed scale for integer frequencies (2^24 guarantees that even the smallest
        # probability ~1e-7 becomes >= 1 after scaling).
        self.SCALE = 2 ** scale_factor

    def _uniform_frequencies(self) -> np.ndarray:
        """Return integer frequencies for a uniform distribution over vocab_size."""
        base = self.SCALE // self.vocab_size
        remainder = self.SCALE - base * self.vocab_size
        freqs = np.full(self.vocab_size, base, dtype=np.int32)
        freqs[:remainder] += 1  # distribute the excess
        return freqs

    @staticmethod
    def _probabilities_to_frequencies(probs: np.ndarray, scale: int) -> np.ndarray:
        """
        Convert float probabilities (sum == 1.0) to integer frequencies (sum == scale)
        such that every non‑zero probability gets a frequency >= 1.
        """
        scaled = probs * scale
        floored = np.floor(scaled).astype(np.int64)
        remainder = scale - np.sum(floored)

        # Distribute remainder to symbols with the largest fractional parts
        fractions = scaled - floored
        indices = np.argpartition(-fractions, remainder)[:remainder]
        floored[indices] += 1

        return floored.astype(np.int32)

    def _compress_chunk(self, hex_chunk: str):
        tokens = self.tokenizer.encode(hex_chunk)
        token_count = len(tokens)
        if token_count == 0:
            return np.array([], dtype=np.uint32), 0

        # Используем буфер для накопления символов
        # Чем больше буфер, тем выше эффективность, но больше задержка
        buffer_size = 64
        symbol_buffer = []
        # Модель для буфера может быть общей, если вероятности не меняются.
        # Но в нашем RNN случае они меняются, поэтому мы будем кодировать буфер
        # каждый раз, как он накопится.

        encoder = constriction.stream.queue.RangeEncoder()
        state = self.model.init_rnn_state(1)

        # Обрабатываем первый токен отдельно (без модели)
        first_token = tokens[0]
        prev_token = torch.tensor([[first_token]], dtype=torch.long, device='cuda')
        _, state = self.model.forward_one_step(prev_token, state)

        # Начинаем накопление со второго токена
        for i in tqdm.tqdm(range(1, token_count)):
            current_token = tokens[i]
            # Получаем вероятности для *текущего* токена на основе предыдущего
            with torch.no_grad():
                logits, state = self.model.forward_one_step(prev_token, state)
                probs = normalize_probabilities(logits[0, 0, :], temperature=1.0)
            prob_np = probs.cpu().to(torch.float64).numpy().astype(np.float64)

            # Создаем модель для текущего символа
            # note: perfect=False может дать небольшой выигрыш в скорости, но perfect=True точнее
            current_model = constriction.stream.model.Categorical(prob_np, perfect=False)

            # Кодируем один символ. encode ожидает массив.
            encoder.encode(current_token, current_model)

            prev_token = torch.tensor([[current_token]], dtype=torch.long, device='cuda')

        # После цикла, когда все символы закодированы, не забываем получить результат
        compressed = encoder.get_compressed()

        # Формируем итоговый массив: [first_token, ...сжатые данные...]
        # Обратите внимание: compressed - это массив uint32
        full = np.concatenate([np.array([first_token], dtype=np.uint32), compressed])
        return full, token_count

    def _decompress_chunk(self, data: np.ndarray, token_count: int) -> str:
        if token_count == 0:
            return ""
        if token_count == 1:
            return self.tokenizer.decode([data[0]])

        # Читаем первый токен
        first_token = data[0]
        # Все остальное - это сжатые данные для кодера
        compressed_data = data[1:]

        decoder = constriction.stream.queue.RangeDecoder(compressed_data)
        state = self.model.init_rnn_state(1)
        prev_token = torch.tensor([[first_token]], dtype=torch.long, device='cuda')
        _, state = self.model.forward_one_step(prev_token, state)

        # Запускаем цикл по всем "сжатым" символам
        reconstructed = [first_token]
        for _ in range(token_count - 1):
            with torch.no_grad():
                logits, state = self.model.forward_one_step(prev_token, state)
                probs = normalize_probabilities(logits[0, 0, :], temperature=1.0)
            prob_np = probs.cpu().to(torch.float64).numpy().astype(np.float64)
            current_model = constriction.stream.model.Categorical(prob_np, perfect=False)

            # Декодируем один символ. decode возвращает массив.
            # Указываем, что нужно декодировать 1 символ
            decoded_arr = decoder.decode(current_model, 1)
            sym = decoded_arr[0]

            reconstructed.append(sym)
            prev_token = torch.tensor([[sym]], dtype=torch.long, device='cuda')

        return self.tokenizer.decode(reconstructed)

    # ------------------------------------------------------------------
    #  Public API – same interface as the original Engine
    # ------------------------------------------------------------------
    @staticmethod
    def prepare_data_to_save(data: np.ndarray):
        """Pack compressed array with a length header (legacy helper)."""
        header = struct.pack('<I', len(data))
        data_bytes = data.astype('<u4').tobytes()
        return header, data_bytes

    @staticmethod
    def read_data(data: bytes) -> np.ndarray:
        """Read compressed array from bytes (legacy helper)."""
        return np.frombuffer(data, dtype='<u4')

    def compress(self, hex_string: str) -> str:
        """
        Compress a hex string representing the original file.
        Splits into chunks of `chunk_size` bytes, compresses each,
        returns a hex string containing the archive.
        """
        raw_bytes = bytes.fromhex(hex_string)
        chunks = [raw_bytes[i:i + self.chunk_size] for i in range(0, len(raw_bytes), self.chunk_size)]
        num_chunks = len(chunks)

        output = bytearray()
        output.extend(struct.pack('<I', num_chunks))

        for chunk in chunks:
            hex_chunk = chunk.hex()
            compressed_arr, token_count = self._compress_chunk(hex_chunk)

            compressed_bytes = compressed_arr.astype('<u4').tobytes()
            orig_len = len(chunk)
            comp_len = len(compressed_bytes)

            output.extend(struct.pack('<III', orig_len, token_count, comp_len))
            output.extend(compressed_bytes)

        return output.hex()

    def decompress(self, compressed_hex: str) -> str:
        """
        Decompress a hex string produced by `compress()` back to the
        original hex string.
        """
        buffer = bytes.fromhex(compressed_hex)
        offset = 0

        num_chunks = struct.unpack_from('<I', buffer, offset)[0]
        offset += 4

        result_bytes = bytearray()

        for _ in range(num_chunks):
            orig_len, token_count, comp_len = struct.unpack_from('<III', buffer, offset)
            offset += 12
            compressed_bytes = buffer[offset:offset + comp_len]
            offset += comp_len

            compressed_arr = self.read_data(compressed_bytes)
            hex_chunk = self._decompress_chunk(compressed_arr, token_count)
            chunk_bytes = bytes.fromhex(hex_chunk)[:orig_len]
            result_bytes.extend(chunk_bytes)

        return result_bytes.hex()
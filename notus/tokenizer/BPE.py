from collections import Counter, deque
from functools import lru_cache
import json
from tqdm import tqdm


class BPETokenizerSimple:
    """A simplified Byte Pair Encoding (BPE) tokenizer implementation."""

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.merge_priority = []  # Хранить порядок слияний для применения
        self.special_tokens = {"<unk>", "<sos>", "<mask>", "<pad>", "<eos>", "<eof>", "<sof>", "<t1>", "<t2>", "<t3>", "<t4>"}

    def apply_bpe(self, token_ids):
        """Применяет BPE-мержи к последовательности токенов"""
        # Проверка на неизвестные токены
        if any(tid not in self.vocab for tid in token_ids):
            unknown = [tid for tid in token_ids if tid not in self.vocab]
            raise ValueError(f"Unknown token IDs: {unknown}")

        # Применяем мержи, пока возможно
        changed = True
        while changed and len(token_ids) > 1:
            changed = False
            new_tokens = []
            i = 0

            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])

                if pair in self.bpe_merges:
                    new_tokens.append(self.bpe_merges[pair])
                    i += 2  # Пропускаем объединенную пару
                    changed = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            # Добавляем последний токен
            if i < len(token_ids):
                new_tokens.append(token_ids[i])

            token_ids = new_tokens

        return token_ids

    def train(self, text, vocab_size, allowed_special=None):
        # Инициализация порядка слияний
        if allowed_special is None:
            allowed_special = {"<unk>", "<sos>", "<mask>", "<pad>", "<eos>", "<eof>", "<sof>", "<t1>", "<t2>", "<t3>", "<t4>"}
        self.merge_priority = []

        # Обработка текста
        processed_text = text
        unique_chars = sorted(set(processed_text))

        # Инициализация словаря
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Добавление специальных токенов
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Конвертация в ID
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # Обучение BPE
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids)
            if pair_id is None:
                break

            # Регистрация слияния
            merged_token = self.vocab[pair_id[0]] + self.vocab[pair_id[1]]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
            self.bpe_merges[pair_id] = new_id
            self.merge_priority.append(pair_id)  # Сохраняем порядок

            # Замена пар
            token_ids = self.replace_pair(token_ids, pair_id, new_id)

    def encode(self, text):
        # Разделение текста на части с учетом специальных токенов
        parts = self.split_text_by_special_tokens(text)

        token_ids = []
        for part in parts:
            if part in self.special_tokens:
                # Обработка специального токена
                token_id = self.inverse_vocab.get(part)
                if token_id is not None:
                    token_ids.append(token_id)
                else:
                    token_ids.append(self.handle_unknown(part))
            else:
                # Обработка обычного текста
                for char in part:
                    if char in self.inverse_vocab:
                        token_ids.append(self.inverse_vocab[char])
                    else:
                        token_ids.append(self.handle_unknown(char))

        # Применение BPE
        return self.apply_bpe(token_ids)

    def split_text_by_special_tokens(self, text):
        """Разделяет текст на части, выделяя специальные токены."""
        parts = []
        start = 0
        while start < len(text):
            found = False
            # Поиск самого длинного совпадения специального токена
            for token in sorted(self.special_tokens, key=len, reverse=True):
                end = start + len(token)
                if text.startswith(token, start, end):
                    # Добавление текста до токена
                    if start > 0:
                        parts.append(text[:start])
                    parts.append(token)
                    text = text[end:]
                    found = True
                    break
            if not found:
                start += 1
        if text:
            parts.append(text)
        return parts

    def handle_unknown(self, char):
        """Обработка неизвестных символов."""
        unk_id = self.inverse_vocab.get("<unk>")
        if unk_id is not None:
            return unk_id
        raise ValueError(f"Unknown character: '{char}' and no <unk> token")

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        # Сохранение словаря
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in self.vocab.items()},
                      f, ensure_ascii=False, indent=2)

        # Сохранение правил слияния с порядком
        with open(bpe_merges_path, "w", encoding="utf-8") as f:
            merges_list = [
                {"pair": list(pair), "new_id": self.bpe_merges[pair]}
                for pair in self.merge_priority
            ]
            json.dump(merges_list, f, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path='.vocab.json', bpe_merges_path='.merges.txt'):
        # Загрузка словаря
        with open(vocab_path, "r", encoding="utf-8") as f:
            loaded_vocab = json.load(f)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Загрузка правил слияния с сохранением порядка
        self.merge_priority = []
        self.bpe_merges = {}
        with open(bpe_merges_path, "r", encoding="utf-8") as f:
            merges_list = json.load(f)
            for merge in merges_list:
                pair = tuple(merge['pair'])
                new_id = merge['new_id']
                self.merge_priority.append(pair)
                self.bpe_merges[pair] = new_id

    def decode(self, token_ids):
        """
        Convert token IDs back to text string.

        Args:
            token_ids: List of token IDs or list of lists (batch) of token IDs

        Returns:
            Decoded text string
        """
        if isinstance(token_ids[0], list):  # Batch processing
            decoded_string = ""
            for sequence in token_ids:
                decoded_string += self.decode(sequence)  # Recursive call
            return decoded_string

        decoded_string = ""
        for token_id in token_ids:
            token = self.vocab[token_id]
            decoded_string += token
        return decoded_string

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        """Get ID for special token with caching."""
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        """
        Find most/least frequent adjacent token pair.

        Args:
            token_ids: List of token IDs
            mode: 'most' for frequent, 'least' for rare

        Returns:
            (token_id1, token_id2) pair or None
        """
        # Count occurrences of adjacent pairs
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Mode must be 'most' or 'least'")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        """
        Replace all occurrences of token pair with new token ID.

        Args:
            token_ids: Original token ID sequence
            pair_id: (id1, id2) pair to replace
            new_id: Token ID to insert as replacement

        Returns:
            New token ID sequence with replacements
        """
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            # Check for pair match with next token
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()  # Remove second element of pair
            else:
                replaced.append(current)

        return replaced
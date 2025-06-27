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
        # Добавляем обязательный токен для неизвестных символов
        self.special_tokens = {"<unk>", "<|endoftext|>"}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>", "<unk>"}):
        """
        Train the tokenizer from scratch using BPE algorithm.

        Args:
            text: Raw training text
            vocab_size: Target vocabulary size
            allowed_special: Special tokens to preserve in vocabulary
        """
        # Preprocessing: Replace spaces with 'Ġ' except at start of text
        # This mimics GPT-2's handling of whitespace
        processed_text = text
        # Initialize vocabulary with base characters
        # Start with all 256 ASCII characters as foundation
        unique_chars = sorted(set(processed_text))

        # Build initial vocab mappings
        self.vocab = {i: char for i, char in tqdm(enumerate(unique_chars))}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add special tokens if not already present
        if allowed_special:
            for token in tqdm(allowed_special):
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Convert processed text to initial token IDs
        token_ids = [self.inverse_vocab[char] for char in tqdm(processed_text)]

        # BPE training loop: merge frequent pairs until vocab_size reached
        for new_id in tqdm(range(len(self.vocab), vocab_size)):
            pair_id = self.find_freq_pair(token_ids)
            if pair_id is None:
                break

            # Создаем новый токен и сразу добавляем в словарь
            merged_token = self.vocab[pair_id[0]] + self.vocab[pair_id[1]]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
            self.bpe_merges[pair_id] = new_id

            token_ids = self.replace_pair(token_ids, pair_id, new_id)

        # Build vocabulary for merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained GPT-2 vocabulary and merge rules.

        Args:
            vocab_path: Path to 'encoder.json' vocabulary file
            bpe_merges_path: Path to 'vocab.bpe' merge rules file
        """
        # Load vocabulary file (token -> id mapping)
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # Create id->token and token->id mappings
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        # Load BPE merge rules
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # Skip version header if present
            if lines and lines[0].startswith("#"):
                lines = lines[1:]

            for rank, line in enumerate(lines):
                pair = tuple(line.strip().split())
                if len(pair) != 2:
                    continue  # Skip malformed lines

                token1, token2 = pair
                # Check if both tokens exist in vocabulary
                if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                    merged_token = token1 + token2
                    # Verify merged token exists in vocab
                    if merged_token in self.inverse_vocab:
                        # Record merge rule
                        self.bpe_merges[(
                            self.inverse_vocab[token1],
                            self.inverse_vocab[token2]
                        )] = self.inverse_vocab[merged_token]

    def encode(self, text):
        # Упрощенная обработка текста
        token_ids = []
        for char in text:
            if char in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[char])
            else:
                # Обработка неизвестных символов
                unk_id = self.inverse_vocab.get("<unk>")
                if unk_id is not None:
                    token_ids.append(unk_id)
                else:
                    raise ValueError(f"Unknown character: '{char}' and no <unk> token")

        # Применяем BPE ко всей последовательности
        return self.apply_bpe(token_ids)

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

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """Save vocabulary and merge rules to JSON files."""
        # Save vocabulary (id->token mapping)
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump({k: v for k, v in self.vocab.items()},
                      file, ensure_ascii=False, indent=2)

        # Save merge rules as list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.bpe_merges.items()
            ]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """Load vocabulary and merge rules from JSON files."""
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load merge rules
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge['pair'])
                self.bpe_merges[pair] = merge['new_id']

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
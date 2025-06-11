from collections import Counter, deque
from functools import lru_cache
import json
from tqdm import tqdm


class BPETokenizerSimple:
    """A simplified Byte Pair Encoding (BPE) tokenizer implementation."""

    def __init__(self):
        # Vocabulary mapping: token_id -> token string
        self.vocab = {}
        # Reverse vocabulary mapping: token string -> token_id
        self.inverse_vocab = {}
        # Stores learned BPE merge rules: (token_id1, token_id2) -> merged_token_id
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """
        Train the tokenizer from scratch using BPE algorithm.

        Args:
            text: Raw training text
            vocab_size: Target vocabulary size
            allowed_special: Special tokens to preserve in vocabulary
        """
        # Preprocessing: Replace spaces with 'Ġ' except at start of text
        # This mimics GPT-2's handling of whitespace
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:  # Non-leading spaces become 'Ġ'
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocabulary with base characters
        # Start with all 256 ASCII characters as foundation
        unique_chars = []
        # Add any unique characters from text not in ASCII
        unique_chars.extend(char for char in sorted(set(processed_text))
                            if char not in unique_chars)
        # Ensure space replacement character exists
        if 'Ġ' not in unique_chars:
            unique_chars.append('Ġ')

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
            # Find most frequent adjacent token pair
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:  # Stop if no mergable pairs remain
                break
            # Replace all occurrences of pair with new token
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            # Record merge rule
            self.bpe_merges[pair_id] = new_id

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
        """
        Convert text to sequence of token IDs using BPE rules.

        Args:
            text: Input string to tokenize

        Returns:
            List of token IDs
        """
        tokens = []
        # Preserve newlines as separate tokens
        words = text.replace("\n", " \n ").split()

        # Reconstruct tokens with space markers
        for i, word in enumerate(words):
            if i > 0 and not word.startswith("\n"):
                tokens.append("Ġ" + word)  # Mark word continuation
            else:
                tokens.append(word)

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                # Use existing token if in vocabulary
                token_ids.append(self.inverse_vocab[token])
            else:
                # Apply BPE subword tokenization
                token_ids.extend(self.tokenize_with_bpe(token))

        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Apply BPE merges to an unknown token.

        Args:
            token: String to tokenize

        Returns:
            List of subword token IDs
        """
        # Start with character-level tokenization
        token_ids = [self.inverse_vocab.get(char) for char in token]
        # Validate all characters are known
        if None in token_ids:
            missing = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Unknown characters: {missing}")

        # Apply merge rules greedily until no more merges possible
        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            # Iterate through token pairs
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                # Check if merge rule exists
                if pair in self.bpe_merges:
                    new_tokens.append(self.bpe_merges[pair])
                    i += 2  # Skip merged token
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            # Add last token if exists
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return token_ids

    def decode(self, token_ids):
        """
        Convert token IDs back to text string.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        decoded_string = ""
        for token_id in token_ids:
            token = self.vocab[token_id]
            # Handle space markers
            if token.startswith("Ġ"):
                decoded_string += " " + token[1:]  # Replace Ġ with space
            else:
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
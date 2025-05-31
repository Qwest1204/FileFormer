from collections import Counter, deque
from functools import lru_cache
import json


class BPETokenizerSimple:
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """
        Train the BPE tokenizer from scratch.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """

        # Preprocess: Replace spaces with 'Ġ'
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including 'Ġ' if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]

        # Extend unique_chars with characters from processed_text that are not already included
        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)

        # Optionally, ensure 'Ġ' is included if it is relevant to your text processing
        if 'Ġ' not in unique_chars:
            unique_chars.append('Ġ')

        # Now create the vocab and inverse vocab dictionaries
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:  # No more pairs to merge. Stopping training.
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.

        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe').
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # loaded_vocab maps token_str to token_id
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}  # token_id: token_str
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}  # token_str: token_id

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # Skip header line if present
            if lines and lines[0].startswith("#"):
                lines = lines[1:]

            for rank, line in enumerate(lines):
                pair = tuple(line.strip().split())
                if len(pair) != 2:
                    print(f"Line {rank+1} has more than 2 entries: {line.strip()}")
                    continue
                token1, token2 = pair
                if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                    token_id1 = self.inverse_vocab[token1]
                    token_id2 = self.inverse_vocab[token2]
                    merged_token = token1 + token2
                    if merged_token in self.inverse_vocab:
                        merged_token_id = self.inverse_vocab[merged_token]
                        self.bpe_merges[(token_id1, token_id2)] = merged_token_id
                        # print(f"Loaded merge: '{token1}' + '{token2}' -> '{merged_token}' (ID: {merged_token_id})")
                    else:
                        print(f"Merged token '{merged_token}' not found in vocab. Skipping.")
                else:
                    print(f"Skipping pair {pair} as one of the tokens is not in the vocabulary.")

    def encode(self, text):
        """
        Encode the input text into a list of token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: The list of token IDs.
        """
        tokens = []
        # Split text into tokens, keeping newlines intact
        words = text.replace("\n", " \n ").split()  # Ensure '\n' is treated as a separate token

        for i, word in enumerate(words):
            if i > 0 and not word.startswith("\n"):
                tokens.append("Ġ" + word)  # Add 'Ġ' to words that follow a space or newline
            else:
                tokens.append(word)  # Handle first word or standalone '\n'

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                # token is contained in the vocabulary as is
                token_id = self.inverse_vocab[token]
                token_ids.append(token_id)
            else:
                # Attempt to handle subword tokenization via BPE
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)

        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.bpe_merges:
                    merged_token_id = self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    # Uncomment for educational purposes:
                    # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                    i += 2  # Skip the next token as it's merged
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return token_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_string = ""
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token.startswith("Ġ"):
                # Replace 'Ġ' with a space
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump({k: v for k, v in self.vocab.items()}, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge['pair'])
                new_id = merge['new_id']
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced
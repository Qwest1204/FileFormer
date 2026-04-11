# ╔══════════════════════════════════════════════════════════════════╗
# ║                  Purity Seal of the Omnissiah                    ║
# ║  ❀  Let this code be sanctified, free from error and heresy.     ║
# ║  ❀  By the Motive Force, let no null pointer or segfault arise.  ║
# ║  ❀  The flesh is weak, but the logic endures.                    ║
# ║  ❀  Praise the Machine God. May the binary be true.              ║
# ║                                                                  ║
# ║              (___)                                               ║
# ║              (o o)   <  Ave Deus Mechanicus!                     ║
# ║              /"\                                                 ║
# ║             /_/ \_                                               ║
# ║                                                                  ║
# ║                     ~ Puritas Codicis ~                          ║
# ╚══════════════════════════════════════════════════════════════════╝

from model import FileFormer, ByteLevelTokenizer
import torch
import constriction
import numpy as np
import struct
from utils.utils import normalize_probabilities


class Engine:
    """
    Engine for compression and decompression using a BERT-like masked approach.

    - The input is split into chunks of `chunk_size` bytes.
    - Each chunk is tokenized, then every `keep_every`-th token is kept,
      while others are replaced by a MASK token (first and last tokens are
      always kept). The kept tokens are stored directly.
    - The masked sequence is passed through the model to obtain probability
      distributions for each masked position.
    - The original tokens at masked positions are entropy-coded using those
      distributions (ANS via constriction).
    - The compressed output contains: chunk metadata, the list of kept tokens,
      and the ANS-compressed masked tokens.
    """

    def __init__(
            self,
            seed: int,
            model: FileFormer,
            tokenizer: ByteLevelTokenizer,
            chunk_size: int = 1024,
            mask_token_id: int = 0,      # Adjust to your tokenizer's MASK id
            keep_every: int = 10,         # Keep 1 token every N (e.g., 10 → 90% masked)
            temperature: float = 5.0
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.mask_token_id = mask_token_id
        self.keep_every = keep_every
        self.temperature = temperature
        self.model.eval()

    # ----------------------------------------------------------------------
    # Core compression / decompression for a single chunk (hex string)
    # ----------------------------------------------------------------------
    def _compress_chunk(self, hex_chunk: str):
        """
        Compress a single hex string chunk using masked language modeling.
        Returns:
            compressed_data : np.ndarray of uint32 (ANS stream)
            kept_tokens     : list[int] of token IDs that were NOT masked
            token_count     : total number of tokens in the chunk
        """
        # 1. Tokenize the hex string
        tokens = self.tokenizer.encode(hex_chunk)
        L = len(tokens)

        # 2. Determine positions to keep (no mask)
        keep_positions = set()
        # Always keep first and last
        keep_positions.add(0)
        if L > 1:
            keep_positions.add(L - 1)

        # Keep every keep_every-th token (excluding boundaries)
        for i in range(1, L - 1):
            if i % self.keep_every == 0:
                keep_positions.add(i)

        # 3. Build masked sequence and collect kept tokens
        masked_tokens = tokens.copy()
        kept_tokens_list = []
        for i in range(L):
            if i in keep_positions:
                kept_tokens_list.append(tokens[i])
            else:
                masked_tokens[i] = self.mask_token_id

        # 4. Get probability distributions for all positions from the model
        masked_tensor = torch.tensor(masked_tokens, dtype=torch.long).unsqueeze(0)  # (1, L)
        with torch.no_grad():
            logits = self.model.forward(masked_tensor)  # (1, L, vocab_size)
            probs_all = normalize_probabilities(logits, temperature=self.temperature)

        # 5. Entropy encode the original tokens at masked positions
        encoder = constriction.stream.queue.RangeEncoder()
        for i in range(L):
            if i not in keep_positions:
                prob_dist = probs_all[0, i, :].cpu().numpy().astype(np.float32)
                model = constriction.stream.model.Categorical(prob_dist, perfect=False)
                true_token = tokens[i]
                encoder.encode(true_token, model)

        compressed_array = encoder.get_compressed()  # np.ndarray of uint32
        return compressed_array, kept_tokens_list, L

    def _decompress_chunk(
            self,
            compressed_array: np.ndarray,
            kept_tokens: list,
            token_count: int
    ) -> str:
        """
        Decompress a single chunk.
        The reconstruction uses the same masked sequence as during compression
        and performs a single forward pass through the model.
        """
        L = token_count

        # 1. Reconstruct the positions that were kept during compression
        keep_positions = set()
        keep_positions.add(0)
        if L > 1:
            keep_positions.add(L - 1)
        for i in range(1, L - 1):
            if i % self.keep_every == 0:
                keep_positions.add(i)

        # 2. Build the initial masked sequence using the kept tokens
        #    All masked positions are initially set to mask_token_id
        masked_tokens = [self.mask_token_id] * L
        kept_idx = 0
        for i in range(L):
            if i in keep_positions:
                masked_tokens[i] = kept_tokens[kept_idx]
                kept_idx += 1

        # 3. Single forward pass to obtain exactly the same distributions
        #    as during compression (all masks still present)
        masked_tensor = torch.tensor(masked_tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = self.model.forward(masked_tensor)
            probs_all = normalize_probabilities(logits, temperature=self.temperature)

        # 4. Decode masked tokens in the exact same order as they were encoded
        decoder = constriction.stream.queue.RangeDecoder(compressed_array)
        reconstructed_tokens = masked_tokens[:]  # will be updated in place
        for i in range(L):
            if i not in keep_positions:
                prob_dist = probs_all[0, i, :].cpu().numpy().astype(np.float32)
                model = constriction.stream.model.Categorical(prob_dist, perfect=False)
                sym = decoder.decode(model)
                reconstructed_tokens[i] = sym

        # 5. Convert token IDs back to hex string
        return self.tokenizer.decode(reconstructed_tokens)

    # ----------------------------------------------------------------------
    # Public API: compress / decompress hex strings
    # ----------------------------------------------------------------------
    def compress(self, hex_string: str) -> str:
        """
        Compress a hex string (representing the original file) into a hex string
        archive containing chunked compressed data with BERT masking.
        """
        raw_bytes = bytes.fromhex(hex_string)

        # Split into chunks
        chunks = [
            raw_bytes[i:i + self.chunk_size]
            for i in range(0, len(raw_bytes), self.chunk_size)
        ]
        num_chunks = len(chunks)

        output_buffer = bytearray()
        output_buffer.extend(struct.pack('<I', num_chunks))  # number of chunks

        for chunk in chunks:
            hex_chunk = chunk.hex()
            compressed_array, kept_tokens, token_count = self._compress_chunk(hex_chunk)

            # Convert compressed_array to bytes
            compressed_bytes = compressed_array.astype('<u4').tobytes()

            original_byte_len = len(chunk)
            kept_tokens_len = len(kept_tokens)
            compressed_byte_len = len(compressed_bytes)

            # Write chunk header
            output_buffer.extend(struct.pack(
                '<IIII',
                original_byte_len,
                token_count,
                kept_tokens_len,
                compressed_byte_len
            ))

            # Write kept tokens (use uint16, assuming vocab < 65536)
            for token in kept_tokens:
                output_buffer.extend(struct.pack('<H', token))

            # Write compressed data
            output_buffer.extend(compressed_bytes)

        return output_buffer.hex()

    def decompress(self, compressed_hex: str) -> str:
        """
        Decompress a hex archive produced by `compress()` back to the original hex string.
        """
        buffer = bytes.fromhex(compressed_hex)
        offset = 0

        num_chunks = struct.unpack_from('<I', buffer, offset)[0]
        offset += 4

        result_bytes = bytearray()

        for _ in range(num_chunks):
            original_byte_len, token_count, kept_tokens_len, compressed_byte_len = \
                struct.unpack_from('<IIII', buffer, offset)
            offset += 16

            # Read kept tokens
            kept_tokens = []
            for _ in range(kept_tokens_len):
                token = struct.unpack_from('<H', buffer, offset)[0]
                kept_tokens.append(token)
                offset += 2

            # Read compressed data
            compressed_bytes = buffer[offset:offset + compressed_byte_len]
            offset += compressed_byte_len
            compressed_array = np.frombuffer(compressed_bytes, dtype='<u4')

            # Decompress chunk
            hex_chunk = self._decompress_chunk(
                compressed_array,
                kept_tokens,
                token_count
            )

            # Convert back to bytes and trim to original length
            chunk_bytes = bytes.fromhex(hex_chunk)[:original_byte_len]
            result_bytes.extend(chunk_bytes)

        return result_bytes.hex()
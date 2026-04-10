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

from model import FileFormer, ByteLevelTokenizer
import torch
import constriction
import numpy as np
import struct
from utils.utils import normalize_probabilities


class Engine:
    """
    Engine for compression and decompression using a learned model.
    Input and output are hex strings. The input file is split into chunks
    of size `chunk_size` bytes, each chunk is compressed separately.
    The resulting compressed hex string contains a header with chunk metadata.
    """

    def __init__(self, seed: int, model: FileFormer, tokenizer: ByteLevelTokenizer, chunk_size: int = 1024):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.model.eval()

    def _compress(self, data: str):
        """
        Compress a single hex string chunk.
        Returns:
            compressed_array: np.ndarray of uint32 (compressed data)
            token_count: number of tokens encoded
        """
        # Tokenize the hex string input
        tokens = self.tokenizer.encode(data)
        _tgt = torch.tensor(tokens, dtype=torch.long)
        token_count = len(_tgt)

        message_encoder = constriction.stream.queue.RangeEncoder()
        # Start-of-sequence token (assumed to be 62)
        context = torch.tensor([62], dtype=torch.long)

        for i in range(token_count):
            with torch.no_grad():
                logits = self.model.forward(context.unsqueeze(0))  # (1, seq_len, vocab_size)
                probs = normalize_probabilities(logits[0, -1, :], temperature=5.0)
            prob_np = probs.cpu().numpy().astype(np.float32)
            model = constriction.stream.model.Categorical(prob_np, perfect=False)

            sym = _tgt[i].item()
            message_encoder.encode(sym, model)

            # Update context with the actual token
            context = torch.cat([context, torch.tensor([sym])])

        return message_encoder.get_compressed(), token_count

    def _decompress(self, data: np.ndarray, len_tgt: int) -> str:
        """
        Decompress a single compressed array back to a hex string.
        """
        message_decoder = constriction.stream.queue.RangeDecoder(data)
        context = torch.tensor([62], dtype=torch.long)
        reconstructed = []

        for _ in range(len_tgt):
            with torch.no_grad():
                logits = self.model.forward(context.unsqueeze(0))
                probs = normalize_probabilities(logits[0, -1, :], temperature=5.0)
            prob_np = probs.cpu().numpy().astype(np.float32)
            model = constriction.stream.model.Categorical(prob_np, perfect=False)

            sym = message_decoder.decode(model)
            reconstructed.append(sym)
            context = torch.cat([context, torch.tensor([sym])])

        return self.tokenizer.decode(reconstructed)

    @staticmethod
    def prepare_data_to_save(data: np.ndarray):
        """Legacy helper: packs compressed array with a length header."""
        header = struct.pack('<I', len(data))
        data_bytes = data.astype('<u4').tobytes()
        return header, data_bytes

    @staticmethod
    def read_data(data: bytes) -> np.ndarray:
        """Legacy helper: reads a compressed array from bytes."""
        return np.frombuffer(data, dtype='<u4')

    def compress(self, hex_string: str) -> str:
        """
        Compress a hex string representing the original file.
        Splits the underlying bytes into chunks of size `self.chunk_size`,
        compresses each chunk, and returns a hex string containing the
        full compressed archive.
        """
        # Convert hex string to raw bytes
        raw_bytes = bytes.fromhex(hex_string)

        # Split into chunks
        chunks = [
            raw_bytes[i:i + self.chunk_size]
            for i in range(0, len(raw_bytes), self.chunk_size)
        ]
        num_chunks = len(chunks)

        # Build output buffer
        output_buffer = bytearray()
        output_buffer.extend(struct.pack('<I', num_chunks))  # number of chunks

        for chunk in chunks:
            # Convert chunk bytes to hex string for tokenizer
            hex_chunk = chunk.hex()
            compressed_array, token_count = self._compress(hex_chunk)

            compressed_bytes = compressed_array.astype('<u4').tobytes()
            original_byte_len = len(chunk)
            compressed_byte_len = len(compressed_bytes)

            # Write chunk metadata
            output_buffer.extend(struct.pack(
                '<III',
                original_byte_len,
                token_count,
                compressed_byte_len
            ))
            # Write compressed data
            output_buffer.extend(compressed_bytes)

        # Return as hex string
        return output_buffer.hex()

    def decompress(self, compressed_hex: str) -> str:
        """
        Decompress a hex string produced by `compress()` back to the
        original hex string.
        """
        buffer = bytes.fromhex(compressed_hex)
        offset = 0

        # Read number of chunks
        num_chunks = struct.unpack_from('<I', buffer, offset)[0]
        offset += 4

        result_bytes = bytearray()

        for _ in range(num_chunks):
            # Read chunk metadata
            original_byte_len, token_count, compressed_byte_len = struct.unpack_from(
                '<III', buffer, offset
            )
            offset += 12

            # Extract compressed data for this chunk
            compressed_bytes = buffer[offset:offset + compressed_byte_len]
            offset += compressed_byte_len

            # Convert back to uint32 array
            compressed_array = self.read_data(compressed_bytes)

            # Decompress to hex string
            hex_chunk = self._decompress(compressed_array, token_count)

            # Convert hex string to bytes and trim to original length
            chunk_bytes = bytes.fromhex(hex_chunk)[:original_byte_len]
            result_bytes.extend(chunk_bytes)

        # Return as hex string
        return result_bytes.hex()
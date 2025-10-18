from typing import Dict, List, Any
from notus.tokenizer import ByteLevelTokenizer

import hashlib
import torch


class CompressionEngine:
    def __init__(self, config: Dict[str, Any], model, device: str = 'cpu'):
        """
        Initialize the CompressionEngine.

        :param config: Configuration dictionary containing 'rule' and 'chunk_size'.
        :param model: Model with encoder and decoder attributes.
        :param device: Device to use for computations (e.g., 'cpu', 'cuda', 'mps').
        """
        self.config = config
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.tokenizer = ByteLevelTokenizer()
        self.device = device
        self.apply_config()

        self.encoder.eval().to(self.device)
        self.decoder.eval().to(self.device)

    def update_config(self, config: Dict[str, Any]):
        """
        Update the configuration and apply it.

        :param config: New configuration dictionary.
        """
        self.config = config
        self.apply_config()

    def apply_config(self):
        """Apply configuration settings. Currently a placeholder for future extensions."""
        # Placeholder for applying config to model or other components if needed
        pass

    def compress(self, chunk: bytes) -> Dict[str, str]:
        """
        Compress the given byte chunk using rule-based slicing.

        :param chunk: Input bytes to compress.
        :return: Dictionary with 'hash' (SHA256 hex) and 'data' (compressed string).
        """
        data_hex = chunk.hex()
        data_hash = hashlib.sha256(chunk).hexdigest()

        encoded_data = self.tokenizer.encode(data_hex)
        compressed_tokens = self._compress_tokens(encoded_data)
        compressed_data = self.tokenizer.decode(compressed_tokens)

        return {"hash": data_hash, "data": compressed_data}

    def decompress(self, data_hash: str, compressed_data: str, file_type: List[int]) -> torch.Tensor:
        """
        Decompress the data by restoring structure and using the model to repair.

        :param data_hash: SHA256 hash hex string of the original data.
        :param compressed_data: Compressed data string.
        :param file_type: List of integers representing file type tokens.
        :return: Repaired data tensor.
        """
        encoded_compressed = self.tokenizer.encode(compressed_data)
        restored_data = self._restore_tokens(encoded_compressed, mask_token=0)

        encoded_hash = self.tokenizer.encode(data_hash)

        tensor_hash = torch.tensor(encoded_hash).unsqueeze(0).to(self.device)
        tensor_type = torch.tensor(file_type).unsqueeze(0).to(self.device)
        tensor_data = torch.tensor(restored_data).unsqueeze(0).to(self.device)

        with torch.no_grad():
            conditioning = self.encoder(tensor_hash, tensor_type)
            repaired_data = self.decoder(tensor_data, conditioning)

        return repaired_data

    @staticmethod
    def test_precision() -> float:
        """
        Test the precision of the compression/decompression process.

        :return: Precision score (placeholder).
        """
        # TODO: Implement actual precision testing logic
        return 0.0

    def _compress_tokens(self, tokens: List[int]) -> List[int]:
        """
        Compress tokens by keeping and cutting segments based on the config rule.

        :param tokens: List of token integers.
        :return: List of kept tokens.
        """
        rule = self.config["rule"]
        a = rule // 100
        b = (rule // 10) % 10
        c = rule % 10
        kept = []
        i = 0
        length = len(tokens)
        while i < length:
            keep_size = min(a * c, length - i)
            kept.extend(tokens[i:i + keep_size])
            i += keep_size
            if i >= length:
                break
            cut_size = min(b * c, length - i)
            i += cut_size  # Skip the cut part
        return kept

    def _restore_tokens(self, kept_tokens: List[int], mask_token: int) -> List[int]:
        """
        Restore the token list by inserting mask tokens where data was cut.

        :param kept_tokens: List of kept tokens.
        :param mask_token: Token to use as mask (e.g., 0).
        :return: Restored list with masks inserted.
        """
        rule = self.config["rule"]
        total_len = self.config["chunk_size"]
        a = rule // 100
        b = (rule // 10) % 10
        c = rule % 10
        result = []
        kept_idx = 0

        while len(result) < total_len:
            # Add kept segments
            take_kept = min(a * c, total_len - len(result), len(kept_tokens) - kept_idx)
            result.extend(kept_tokens[kept_idx:kept_idx + take_kept])
            kept_idx += take_kept

            # Add mask segments
            take_mask = min(b * c, total_len - len(result))
            result.extend([mask_token] * take_mask)

        # Truncate if exceeded (safety)
        return result[:total_len]
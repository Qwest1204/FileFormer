from typing import Dict, List, Any, Tuple
import hashlib
import torch
from tqdm import tqdm

from notus.tokenizer import ByteLevelTokenizer  # Assuming this is a custom or specific tokenizer


class CompressionEngine:
    def __init__(self, config: Dict[str, Any], model: Any, device: str = 'cpu'):
        """
        Initialize the CompressionEngine.

        :param config: Configuration dictionary containing 'rule', 'chunk_size', optionally 'mask_token' and 'file_type'.
        :param model: Model that performs forward pass for repair (assumed to have .forward method).
        :param device: Device for computations ('cpu', 'cuda', 'mps', etc.).
        """
        self._validate_config(config)
        self.config = config
        self.model = model.to(device)
        self.tokenizer = ByteLevelTokenizer()
        self.device = device
        self.mask_token = config.get('mask_token', 0)
        self.file_type = config.get('file_type', [487])
        self._parse_rule()

    def _validate_config(self, config: Dict[str, Any]) -> None:
        required_keys = ['rule', 'chunk_size']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config missing required key: {key}")
        if config['rule'] < 0:
            raise ValueError("Rule must be a non-negative integer.")

    def _parse_rule(self) -> None:
        """Parse the compression rule into components for reuse."""
        rule = self.config["rule"]
        self.keep_multiplier = rule // 100  # a
        self.skip_multiplier = (rule // 10) % 10  # b
        self.base = rule % 10  # c
        if any(x < 0 for x in [self.keep_multiplier, self.skip_multiplier, self.base]):
            raise ValueError("Parsed rule components must be non-negative.")

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration and reapply settings.

        :param config: New configuration dictionary.
        """
        self._validate_config(config)
        self.config = config
        self.mask_token = config.get('mask_token', 0)
        self.file_type = config.get('file_type', [487])
        self._parse_rule()

    def full_file_compress(self, file_path: str, meta_seek_from_start: int, meta_seek_from_end: int, chunk_size: int = None) -> Dict[str, Any]:
        """
        Compress the entire file by processing metadata and data chunks.

        :param file_path: Path to the input file.
        :param meta_seek_from_start: Bytes to read as starting metadata.
        :param meta_seek_from_end: Bytes to read as ending metadata.
        :param chunk_size: Size of each data chunk in bytes (defaults to config['chunk_size']).
        :return: Dictionary with compressed data, hashes, lengths, and metadata.
        """
        if chunk_size is None:
            chunk_size = self.config['chunk_size']

        compressed_data = []
        hashes = []
        hex_lengths = []
        original_token_lengths = []
        start_metadata_hex = ''
        end_metadata_hex = ''

        try:
            with open(file_path, 'rb') as file:
                start_metadata_hex = file.read(meta_seek_from_start).hex()

                if meta_seek_from_end > 0:
                    file.seek(-meta_seek_from_end, 2)  # Seek from end
                    end_metadata_hex = file.read(meta_seek_from_end).hex()
                else:
                    file.seek(meta_seek_from_start)  # Reset to after start meta for chunks

                for _ in tqdm(range(0, file.seek(0, 2) - meta_seek_from_start - meta_seek_from_end, chunk_size)):
                    file.seek(meta_seek_from_start + _)
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break

                    hex_chunk = chunk.hex()
                    chunk_hash = hashlib.sha256(chunk).hexdigest()

                    compressed_str, orig_len = self.compress(hex_chunk)
                    compressed_data.append(compressed_str)
                    hashes.append(chunk_hash)
                    hex_lengths.append(len(hex_chunk))
                    original_token_lengths.append(orig_len)

        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Error reading file {file_path}: {e}")

        return {
            'mdata_start': start_metadata_hex,
            'mdata_end': end_metadata_hex,
            'data': compressed_data,
            'lens': hex_lengths,
            'hashs': hashes,
            'orig_token_lens': original_token_lengths,
        }

    def full_file_decompress(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Decompress the entire file data from the compressed dictionary.

        :param input_data: Dictionary from compression containing 'data', 'hashs', 'lens', etc.
        :return: List of decompressed hex strings for each chunk.
        """
        decompressed_hex_chunks = []
        num_chunks = len(input_data['data'])

        for i in tqdm(range(num_chunks)):
            repaired_tokens = self.decompress(
                data_hash=input_data['hashs'][i],
                compressed_data=input_data['data'][i],
                file_type=self.file_type,
                original_token_len=input_data['orig_token_lens'][i]
            )
            decompressed_hex = self.tokenizer.decode(repaired_tokens)

            # Truncate all chunks to original hex length
            original_hex_len = input_data['lens'][i]
            if len(decompressed_hex) < original_hex_len:
                raise ValueError(f"Decompressed hex for chunk {i} is shorter than original: {len(decompressed_hex)} < {original_hex_len}")
            decompressed_hex = decompressed_hex[:original_hex_len]

            decompressed_hex_chunks.append(decompressed_hex)

        return decompressed_hex_chunks

    def write_decompressed_to_file(self, input_data: Dict[str, Any], output_path: str) -> None:
        """
        Decompress the data and write the metadata followed by the decompressed bytes to a file.

        :param input_data: Dictionary from compression containing 'mdata_start', 'mdata_end', 'data', etc.
        :param output_path: Path to the output file where decompressed data will be written.
        """
        try:
            decompressed_hex_chunks = self.full_file_decompress(input_data)

            with open(output_path, 'wb') as file:
                if 'mdata_start' in input_data and input_data['mdata_start']:
                    meta_bytes = bytes.fromhex(input_data['mdata_start'])
                    file.write(meta_bytes)

                for hex_str in decompressed_hex_chunks:
                    chunk_bytes = bytes.fromhex(hex_str)
                    file.write(chunk_bytes)

                if 'mdata_end' in input_data and input_data['mdata_end']:
                    meta_bytes = bytes.fromhex(input_data['mdata_end'])
                    file.write(meta_bytes)

        except (IOError, ValueError) as e:
            raise ValueError(f"Error writing to file {output_path}: {e}")

    def verify_decompression(self, input_data: Dict[str, Any], output_path: str) -> bool:
        """
        Verify the decompressed file by recomputing hashes and comparing to originals.

        :param input_data: Original compressed dictionary.
        :param output_path: Path to the decompressed file.
        :return: True if all chunk hashes match, False otherwise.
        """
        chunk_size = self.config['chunk_size']
        try:
            with open(output_path, 'rb') as file:
                if 'mdata_start' in input_data:
                    file.read(len(input_data['mdata_start']) // 2)  # Skip start meta bytes

                for i, orig_hash in enumerate(input_data['hashs']):
                    chunk = file.read(min(chunk_size, input_data['lens'][i] // 2))
                    recomputed_hash = hashlib.sha256(chunk).hexdigest()
                    if recomputed_hash != orig_hash:
                        return False

            return True
        except (FileNotFoundError, IOError):
            return False

    def compress(self, hex_data: str) -> Tuple[str, int]:
        """
        Compress a hex string chunk using tokenization and rule-based slicing.

        :param hex_data: Input hex string to compress.
        :return: Compressed string and original token length.
        """
        encoded_tokens = self.tokenizer.encode(hex_data)
        original_token_len = len(encoded_tokens)
        compressed_tokens = self._compress_tokens(encoded_tokens)
        compressed_str = self.tokenizer.decode(compressed_tokens)
        return compressed_str, original_token_len

    def decompress(self, data_hash: str, compressed_data: str, file_type: List[int], original_token_len: int) -> List[int]:
        """
        Decompress by restoring structure, repairing with model, and inserting predicted parts only into masked positions.

        :param data_hash: SHA256 hash hex string of the original data.
        :param compressed_data: Compressed data string.
        :param file_type: List of integers representing file type tokens.
        :param original_token_len: Original length of the token sequence before compression.
        :return: List of repaired tokens.
        """
        encoded_compressed = self.tokenizer.encode(compressed_data)
        restored_tokens = self._restore_tokens(
            kept_tokens=encoded_compressed,
            mask_token=self.mask_token,
            total_token_len=original_token_len
        )

        encoded_hash = self.tokenizer.encode(data_hash)

        with torch.no_grad():
            tensor_hash = torch.tensor(encoded_hash).unsqueeze(0).to(self.device)
            tensor_type = torch.tensor(file_type).unsqueeze(0).to(self.device)
            tensor_data = torch.tensor(restored_tokens).unsqueeze(0).to(self.device)
            logits = self.model.forward(tensor_data, tensor_hash, tensor_type)

        predicted_tokens = torch.argmax(logits, dim=-1)[0].tolist()

        # Insert only the predicted parts into the masked positions, keeping originals elsewhere
        repaired_tokens = [
            predicted_tokens[i] if token == self.mask_token else token
            for i, token in enumerate(restored_tokens)
        ]

        return repaired_tokens

    def _compress_tokens(self, tokens: List[int]) -> List[int]:
        """
        Compress tokens by keeping segments and skipping others based on the rule.

        :param tokens: List of token integers.
        :return: List of kept tokens.
        """
        kept_tokens = []
        i = 0
        token_count = len(tokens)
        while i < token_count:
            keep_size = min(self.keep_multiplier * self.base, token_count - i)
            kept_tokens.extend(tokens[i:i + keep_size])
            i += keep_size

            if i >= token_count:
                break

            skip_size = min(self.skip_multiplier * self.base, token_count - i)
            i += skip_size  # Skip this segment

        return kept_tokens

    def _restore_tokens(self, kept_tokens: List[int], mask_token: int, total_token_len: int) -> List[int]:
        """
        Restore the token list by inserting mask tokens in skipped positions.

        :param kept_tokens: List of kept tokens from compression.
        :param mask_token: Token value to use for masks.
        :param total_token_len: Original total length of the token sequence.
        :return: Restored list with masks inserted, exactly of original length.
        """
        restored = []
        kept_index = 0
        kept_len = len(kept_tokens)

        while len(restored) < total_token_len:
            # Add kept segment
            keep_size = min(self.keep_multiplier * self.base, total_token_len - len(restored), kept_len - kept_index)
            restored.extend(kept_tokens[kept_index:kept_index + keep_size])
            kept_index += keep_size

            # Add mask segment
            skip_size = min(self.skip_multiplier * self.base, total_token_len - len(restored))
            restored.extend([mask_token] * skip_size)

        if len(restored) != total_token_len:
            raise ValueError(f"Restored tokens length mismatch: {len(restored)} != {total_token_len}")
        if kept_index != kept_len:
            raise ValueError(f"Unused kept tokens: used {kept_index}, total {kept_len}")

        return restored
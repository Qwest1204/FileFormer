import zlib
import torch
from tqdm import tqdm
from pathlib import Path
from hashlib import sha256, md5
from tokenizer import Tokenizer
from torch.utils.data import Dataset


class FileDataset(Dataset):
    """PyTorch Dataset for processing files into tokenized chunks with metadata.

    Converts file data into fixed-size chunks, computes hashes, and handles tokenization
    with padding. Designed for training models that process binary data.
    """

    def __init__(self, path, hash_type, max_bytes):
        """Initialize dataset with processing parameters.

        Args:
            path (str/Path): Directory path containing files to process
            hash_type (str): Type of hash to compute - 'md5', 'sha256', or 'crc32'
            max_bytes (int): Maximum chunk size in bytes for file segmentation

        Raises:
            ValueError: If unsupported hash_type is provided
        """
        self.tokenizer = Tokenizer()
        self.files = [x for x in Path(path).glob('**/*') if x.is_file()]  # Recursive file listing
        self.max_bytes = max_bytes  # Maximum chunk size per sample

        # Configure hash function
        if hash_type == 'md5':
            self.hash_func = md5
        elif hash_type == 'sha256':
            self.hash_func = sha256
        elif hash_type == 'crc32':
            # Wrap crc32 to ensure consistent 32-bit unsigned output
            self.hash_func = lambda x: zlib.crc32(x) & 0xFFFFFFFF
        else:
            raise ValueError(f"Unsupported hash type: {hash_type}")

        self.data = self.prepare(self.files, self.max_bytes)  # Process all files during init

    def prepare(self, files, max_bytes):
        """Process files into dataset entries with chunking, hashing, and tokenization.

        Args:
            files (list[Path]): List of file paths to process
            max_bytes (int): Maximum size in bytes for each data chunk

        Returns:
            list: Processed dataset entries containing:
                - hash: Hexadecimal hash string of chunk
                - hex_data: Raw hex representation of chunk
                - tokens: Padded token sequence for model input

        Note:
            - Files smaller than max_bytes will produce single chunks
            - Chunks exactly divisible by max_bytes won't be padded
            - Final chunk of oversized files may be smaller than max_bytes
        """
        data = []
        for file in tqdm(files, desc="Processing files"):
            with open(file, 'rb') as f:
                data_from_file = f.read()
                # Split file into max_bytes chunks
                chunks = [
                    data_from_file[i:i + max_bytes]
                    for i in range(0, len(data_from_file), max_bytes)
                ]

            for chunk in chunks:
                # Compute hash with type-specific handling
                hash_obj = self.hash_func(chunk)
                hash_str = (f"{hash_obj:08x}" if isinstance(hash_obj, int)
                            else hash_obj.hexdigest())

                # Tokenize and pad sequence
                tokens = self.tokenizer.encode(chunk)
                target_token_length = max_bytes // 2  # 2 bytes per token baseline

                # Apply padding if needed (does NOT truncate oversize sequences)
                if len(tokens) < target_token_length:
                    pad_size = target_token_length - len(tokens)
                    tokens += [self.tokenizer.get_idx_from_token('<PAD>')] * pad_size

                data.append({
                    'hash': hash_str,  # Hash identifier for chunk
                    'hex_data': chunk.hex(),  # Raw hex representation
                    'tokens': tokens  # Padded token sequence (may exceed target length)
                })
        return data

    def __len__(self):
        """Return total number of processed chunks in dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get dataset item as model-ready format.

        Returns:
            tuple: (token_tensor, hash_string)
                - token_tensor (torch.Tensor): Padded token sequence as tensor
                - hash_string (str): Hexadecimal hash of the chunk

        Note:
            - Actual sequence lengths may vary if encoded tokens (including control tokens)
              exceed max_bytes//2
            - Consumers should handle padding/truncation during batch collation
        """
        item = self.data[idx]
        return torch.tensor(item['tokens']), item['hash']
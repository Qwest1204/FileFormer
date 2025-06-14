from tokenizer import Tokenizer
from transformer_model import Transformer
import torch
from tqdm import tqdm

class CompressEngine:
    """
    Compression engine that uses transformer models to compress and decompress data.
    """
    def __init__(self, tokenizer: Tokenizer, compress_unit: Transformer,
                 error_fix_unit: Transformer, chunk_compress_size=256,):
        """
        Initializes the CompressEngine with a tokenizer, compression unit, error fix unit, and chunk size.

        Args:
            tokenizer: Tokenizer instance for encoding and decoding tokens.
            compress_unit: Transformer model used for compression.
            error_fix_unit: Transformer model used for error correction during decompression.
            chunk_compress_size: Size of chunks for compression.
        """
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_token = self.tokenizer.get_special_token_id("<|PAD|>")
        self.mask_token = self.tokenizer.get_special_token_id("<|UNK|>")
        self.compress_unit = compress_unit.to(self.device)
        self.error_fix_unit = error_fix_unit.to(self.device)
        self.chunk_compress_size = chunk_compress_size

    def apply_mask(self, tensor, mask):
        """
        Applies a mask to a tensor by setting masked positions to the mask token.

        Args:
            tensor: Tensor to which the mask will be applied.
            mask: Mask tensor indicating positions to be masked.

        Returns:
            Tensor with mask applied.
        """
        tensor[mask.bool()] = self.mask_token
        return tensor

    def max_probabilities(self, tensor1, tensor2):
        """
        Compares two tensors and selects the indices from the tensor with higher maximum probabilities.

        Args:
            tensor1: First tensor of probabilities.
            tensor2: Second tensor of probabilities.

        Returns:
            Tensor of indices with higher probabilities.
        """
        max_vals1, max_indices1 = torch.max(tensor1, dim=-1)
        max_vals2, max_indices2 = torch.max(tensor2, dim=-1)
        source_mask = max_vals1 >= max_vals2
        final_indices = torch.where(source_mask, max_indices1, max_indices2)
        return final_indices

    def create_mask(self, tensor1, tensor2):
        """
        Creates a mask based on differences between two tensors, including neighboring positions.

        Args:
            tensor1: First tensor to compare.
            tensor2: Second tensor to compare.

        Returns:
            Mask tensor indicating positions with differences and their neighbors.
        """
        initial_mask = (tensor1 != tensor2).int()
        new_mask = initial_mask.clone()

        indices = torch.where(initial_mask == 1)[0]

        if indices.numel() > 0:
            left_indices = indices - 1
            right_indices = indices + 1

            left_indices = left_indices[left_indices >= 0]
            right_indices = right_indices[right_indices < initial_mask.size(0)]

            all_indices = torch.cat([left_indices, right_indices])
            new_mask[all_indices] = 1

        return new_mask

    def compress(self, data_from_file: bytes):
        """
        Compresses input data by tokenizing, chunking, and applying the compression unit.

        Args:
            data_from_file: Input data as bytes to be compressed.

        Returns:
            Tuple of compressed data and masks.
        """
        tokens = self.tokenizer.encode(data_from_file.hex())
        num_chunks = (len(tokens) + self.chunk_compress_size - 1) // self.chunk_compress_size

        chunks = torch.full((num_chunks, self.chunk_compress_size), self.pad_token, dtype=torch.int32, device=self.device)
        masks = torch.zeros((num_chunks, self.chunk_compress_size), dtype=torch.int32, device=self.device)

        for i in range(num_chunks):
            start = i * self.chunk_compress_size
            end = start + self.chunk_compress_size
            actual_len = min(end, len(tokens)) - start
            chunk_data = tokens[start:end]

            chunks[i, :actual_len] = torch.tensor(chunk_data, dtype=torch.int32, device=self.device)
            masks[i, :actual_len] = 1

        compress_data = []
        maskss = []
        for chunk, mask in tqdm(zip(chunks, masks)):
            compressed_chunk = self.compress_unit.encode(chunk.unsqueeze(0), mask.unsqueeze(0))
            compress_data.append(compressed_chunk.squeeze(0))
            maskss.append(mask)

        return compress_data, maskss

    def decompress(self, data, mask, deep_of_error_correction) -> bytes:
        """
        Decompresses data by decoding, applying error correction, and converting back to bytes.

        Args:
            data: Compressed data.
            mask: Mask associated with the compressed data.
            deep_of_error_correction: Depth of error correction iterations.

        Returns:
            Decompressed data as bytes.
        """
        data = torch.stack(data).to(self.device)
        mask = torch.stack(mask).to(self.device)
        batch_size = data.size(0)

        print(f"Input data shape: {data.shape}")
        print(f"Input mask shape: {mask.shape}")

        decoder_out = self.compress_unit.decode(data, mask)
        print(f"Decoder output shape: {decoder_out.shape}")

        decoder_out_projected = self.compress_unit.project(decoder_out)
        print(f"Projected decoder output shape: {decoder_out_projected.shape}")

        decoder_out_main = torch.argmax(decoder_out_projected, dim=-1)
        decoder_out_alternate = decoder_out_main.clone()

        for _ in range(deep_of_error_correction):
            mask_mask = self.create_mask(decoder_out_main, decoder_out_alternate)
            mark_src = self.apply_mask(decoder_out_main.clone(), mask_mask)

            decoder_out_main_n = self.error_fix_unit.forward(mark_src, mask)
            decoder_out_alternate_n = self.error_fix_unit.forward(mark_src, mask)

            decoder_out_main = torch.argmax(decoder_out_main_n, dim=-1)
            decoder_out_alternate = torch.argmax(decoder_out_alternate_n, dim=-1)

        token_ids = self.max_probabilities(decoder_out_main_n, decoder_out_alternate_n)
        hex_str = self.tokenizer.decode(token_ids.cpu().tolist())

        if len(hex_str) % 2 == 0:
            return bytes.fromhex(hex_str)
        else:
            return bytes.fromhex(hex_str[:-1])

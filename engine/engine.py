from tokenizer import Tokenizer
from transformer_model import Transformer
import torch
from tqdm import tqdm

class CompressEngine:
    def __init__(self, tokenizer: Tokenizer, compress_unit: Transformer,
                 error_fix_unit: Transformer, chunk_compress_size=256,):
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_token = self.tokenizer.get_special_token_id("<|PAD|>")
        self.mask_token = self.tokenizer.get_special_token_id("<|UNK|>")
        self.compress_unit = compress_unit.to(self.device)
        self.error_fix_unit = error_fix_unit.to(self.device)
        self.chunk_compress_size = chunk_compress_size

    def apply_mask(self, tensor, mask):
        tensor[mask.bool()] = self.mask_token
        return tensor

    def max_probabilities(self, tensor1, tensor2):
        max_vals1, max_indices1 = torch.max(tensor1, dim=-1)
        max_vals2, max_indices2 = torch.max(tensor2, dim=-1)
        source_mask = max_vals1 >= max_vals2
        final_indices = torch.where(source_mask, max_indices1, max_indices2)
        return final_indices

    def create_mask(self, tensor1, tensor2):
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



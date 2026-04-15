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
import torch.nn as nn
import torch.optim as optim
import constriction
import numpy as np
import struct
from tqdm import tqdm
import copy
import random
from typing import Optional
from utils.utils import (
    normalize_probabilities,
    freeze_all_except_lora,
    convert_state_dict_to_lora,
    load_model
)
from muon import MuonWithAuxAdam


class Engine:
    """
    Engine for compression and decompression using a BERT-like masked approach.

    - The input is split into chunks of `chunk_size` bytes.
    - For every `lora_per_chanks` chunks, the model's LoRA parameters are
      fine‑tuned on those chunks to improve compression. The adapted LoRA
      weights are stored in the archive and applied during decompression.
    - Each chunk is tokenized, then every `keep_every`-th token is kept,
      while others are replaced by a MASK token (first and last tokens are
      always kept). The kept tokens are stored directly.
    - The masked sequence is passed through the (possibly adapted) model to
      obtain probability distributions for each masked position.
    - The original tokens at masked positions are entropy-coded using those
      distributions (ANS via constriction).
    - The compressed output contains: number of chunks, and for each group of
      `lora_per_chanks` chunks: LoRA weights followed by the chunk data.
    """

    def __init__(
            self,
            seed: int,
            model: FileFormer,
            tokenizer: ByteLevelTokenizer,
            chunk_size: int = 1024,
            mask_token_id: int = 0,      # Adjust to your tokenizer's MASK id
            keep_every: int = 10,         # Keep 1 token every N (e.g., 10 → 90% masked)
            temperature: float = 5.0,
            lora_per_chanks: int = 4,
            lora_finetune_epochs: int = 1,
            lora_learning_rate_muon: float = 0.1,
            lora_learning_rate_adamw: float = 3e-4,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.mask_token_id = mask_token_id
        self.keep_every = keep_every
        self.temperature = temperature
        self.lora_per_chanks = lora_per_chanks
        self.lora_finetune_epochs = lora_finetune_epochs
        self.lora_learning_rate_muon = lora_learning_rate_muon
        self.lora_learning_rate_adamw = lora_learning_rate_adamw
        self.model.eval()

        # Cache the initial LoRA state (if any) to restore after each group
        self._base_lora_state = self._get_lora_state_dict()

    # ----------------------------------------------------------------------
    # LoRA utilities
    # ----------------------------------------------------------------------
    def _get_lora_state_dict(self) -> dict:
        """
        Return a copy of the current LoRA parameters from the model.
        Assumes LoRA parameters are identified by 'lora_' in their names.
        """
        lora_state = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                lora_state[name] = param.data.clone()
        return lora_state

    def _set_lora_state_dict(self, lora_state: dict):
        """
        Load the given LoRA state into the model.
        """
        self.model.load_state_dict(lora_state, strict=False)

    def _reset_lora_to_base(self):
        """Restore the LoRA parameters to the base state saved at init."""
        self._set_lora_state_dict(self._base_lora_state)

    def _serialize_lora_state(self, lora_state: dict) -> bytes:
        """
        Convert a LoRA state dict to a compact byte representation.
        Format: for each parameter: 2 bytes for name length, name (UTF-8),
                4 bytes for number of elements, then raw float32 data.
        """
        buffer = bytearray()
        for name, tensor in sorted(lora_state.items()):
            name_bytes = name.encode('utf-8')
            buffer.extend(struct.pack('<H', len(name_bytes)))
            buffer.extend(name_bytes)
            arr = tensor.cpu().numpy().astype(np.float32)
            buffer.extend(struct.pack('<I', arr.size))
            buffer.extend(arr.tobytes())
        return bytes(buffer)

    def _deserialize_lora_state(self, data: bytes) -> dict:
        """
        Inverse of _serialize_lora_state.
        """
        offset = 0
        lora_state = {}
        while offset < len(data):
            name_len = struct.unpack_from('<H', data, offset)[0]
            offset += 2
            name = data[offset:offset+name_len].decode('utf-8')
            offset += name_len
            numel = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            arr = np.frombuffer(data[offset:offset+numel*4], dtype=np.float32)
            offset += numel * 4
            # Reconstruct tensor (shape will be inferred when loading into model)
            lora_state[name] = torch.from_numpy(arr.copy())
        return lora_state

    # ----------------------------------------------------------------------
    # LoRA fine‑tuning on a group of chunks
    # ----------------------------------------------------------------------
    def _prepare_lora_finetune(self):
        """
        Freeze all model parameters except those belonging to LoRA.
        Returns an optimizer configured for the LoRA parameters.
        """
        muon_params = []
        adamw_params = []
        freeze_all_except_lora(self.model)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Muon работает только с 2D тензорами (веса линейных слоёв)
            if param.ndim == 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        weight_decay = 0.01
        betas = (0.95, 0.95)

        # Формируем param_groups для MuonWithAuxAdam
        param_groups = [
            {
                'params': muon_params,
                'use_muon': True,
                'lr': self.lora_learning_rate_muon,
                'weight_decay': weight_decay,
                'momentum': 0.95,  # Параметр momentum для Muon
            },
            {
                'params': adamw_params,
                'use_muon': False,
                'lr': self.lora_learning_rate_adamw,
                'betas': betas,  # betas используются только для AdamW
                'weight_decay': weight_decay,
            }
        ]

        # Создаём и возвращаем оптимизатор
        optimizer = torch.optim.Muon(param_groups)
        return optimizer

    def _finetune_on_chunks(self, chunks_data: list) -> dict:
        """
        Fine‑tune LoRA на группе чанков.
        Процент маскируемых токенов растёт с каждой эпохой после второй:
        первые 2 эпохи – 0%, далее +7.5% за эпоху.
        Loss вычисляется по всем позициям (включая незамаскированные).
        """
        if not chunks_data:
            return self._get_lora_state_dict()

        self.model.train()
        optimizer = self._prepare_lora_finetune()
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.mask_token_id)  # игнорируем паддинги

        for epoch in range(self.lora_finetune_epochs):
            # Определяем процент удаления для текущей эпохи
            if epoch < 2:
                p_mask = 0.0
            else:
                p_mask = 0.075 * (epoch - 1)
                if p_mask > 1.0:
                    p_mask = 1.0

            total_loss = 0.0
            batch_inputs = []  # замаскированные последовательности
            batch_targets = []  # оригинальные последовательности (полные)

            for chunk_bytes in chunks_data:
                hex_chunk = chunk_bytes.hex()
                tokens = self.tokenizer.encode(hex_chunk)
                L = len(tokens)
                if L <= 2:
                    continue

                # Копируем оригинальные токены
                masked_tokens = tokens.copy()

                # Маскируем выбранные позиции (исключая первый и последний)
                if p_mask > 0.0:
                    for i in range(1, L - 1):
                        if random.random() < p_mask:
                            masked_tokens[i] = self.mask_token_id

                # Сохраняем вход (маскированный) и полный target (оригинал)
                batch_inputs.append(masked_tokens)
                batch_targets.append(tokens.copy())  # оригинал целиком

            if not batch_inputs:
                continue

            # Паддинг до максимальной длины в батче
            max_len = max(len(seq) for seq in batch_inputs)
            padded_inputs = []
            padded_targets = []
            attention_mask = []  # может пригодиться, но loss_fn использует ignore_index

            for input_seq, target_seq in zip(batch_inputs, batch_targets):
                pad_len = max_len - len(input_seq)
                # Для входов паддинг заполняем mask_token_id (можно и pad_token_id)
                padded_inputs.append(input_seq + [self.mask_token_id] * pad_len)
                # Для target паддинг заполняем mask_token_id, чтобы loss их игнорировал
                padded_targets.append(target_seq + [self.mask_token_id] * pad_len)
                attention_mask.append([1] * len(input_seq) + [0] * pad_len)

            input_tensor = torch.tensor(padded_inputs, dtype=torch.long)
            target_tensor = torch.tensor(padded_targets, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool)

            optimizer.zero_grad()
            logits = self.model.forward(input_tensor)  # (B, max_len, vocab_size)

            # Вычисляем loss для всех позиций (игнорируем pad-токены через ignore_index)
            # logits: (B, max_len, vocab_size), target_tensor: (B, max_len)
            loss = loss_fn(logits.permute(0, 2, 1), target_tensor)  # CrossEntropy ожидает (N, C, ...)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if self.lora_finetune_epochs > 1:
                print(f"Epoch {epoch + 1}/{self.lora_finetune_epochs} loss: {total_loss:.4f}, p_mask={p_mask:.3f}")

        self.model.eval()
        return self._get_lora_state_dict()

    # ----------------------------------------------------------------------
    # Core compression / decompression for a single chunk
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
        masked_tokens = [self.mask_token_id] * L
        kept_idx = 0
        for i in range(L):
            if i in keep_positions:
                masked_tokens[i] = kept_tokens[kept_idx]
                kept_idx += 1

        # 3. Single forward pass to obtain exactly the same distributions
        masked_tensor = torch.tensor(masked_tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = self.model.forward(masked_tensor)
            probs_all = normalize_probabilities(logits, temperature=self.temperature)

        # 4. Decode masked tokens in the exact same order as they were encoded
        decoder = constriction.stream.queue.RangeDecoder(compressed_array)
        reconstructed_tokens = masked_tokens[:]
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
        archive containing chunked compressed data with per‑group LoRA adaptation.
        """
        raw_bytes = bytes.fromhex(hex_string)

        # Split into chunks
        chunks = [
            raw_bytes[i:i + self.chunk_size]
            for i in range(0, len(raw_bytes), self.chunk_size)
        ]
        num_chunks = len(chunks)

        output_buffer = bytearray()
        output_buffer.extend(struct.pack('<I', num_chunks))  # total number of chunks

        # Process in groups of lora_per_chanks
        for group_start in tqdm(range(0, num_chunks, self.lora_per_chanks)):
            group_end = min(group_start + self.lora_per_chanks, num_chunks)
            group_chunks = chunks[group_start:group_end]

            # Reset LoRA to base state before fine‑tuning on this group
            self._reset_lora_to_base()

            # Fine‑tune LoRA on this group
            adapted_lora_state = self._finetune_on_chunks(group_chunks)
            # Serialize and write LoRA state for this group
            lora_bytes = self._serialize_lora_state(adapted_lora_state)
            output_buffer.extend(struct.pack('<I', len(lora_bytes)))
            output_buffer.extend(lora_bytes)

            # Compress each chunk in the group using the adapted model
            for chunk in group_chunks:
                hex_chunk = chunk.hex()
                compressed_array, kept_tokens, token_count = self._compress_chunk(hex_chunk)

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

                # Write kept tokens (uint16)
                for token in kept_tokens:
                    output_buffer.extend(struct.pack('<H', token))

                # Write compressed data
                output_buffer.extend(compressed_bytes)

        # Restore base LoRA state after compression (cleanup)
        self._reset_lora_to_base()
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
        chunks_processed = 0

        while chunks_processed < num_chunks:
            # Read LoRA state length and data for this group
            lora_len = struct.unpack_from('<I', buffer, offset)[0]
            offset += 4
            lora_bytes = buffer[offset:offset + lora_len]
            offset += lora_len

            # Deserialize and apply LoRA state
            lora_state = self._deserialize_lora_state(lora_bytes)
            fixed_state_dict = {}
            for key, value in lora_state.items():
                if 'lora_A' in key and value.dim() == 1:
                    # Предполагаем, что ранг r = 4 (из сообщения об ошибке)
                    r = 4
                    in_features = value.numel() // r
                    fixed_state_dict[key] = value.view(r, in_features)
                elif 'lora_B' in key and value.dim() == 1:
                    r = 4
                    out_features = value.numel() // r
                    fixed_state_dict[key] = value.view(out_features, r)
                else:
                    fixed_state_dict[key] = value
            self._set_lora_state_dict(fixed_state_dict)

            # Determine how many chunks in this group
            group_size = min(self.lora_per_chanks, num_chunks - chunks_processed)

            for _ in range(group_size):
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

                chunk_bytes = bytes.fromhex(hex_chunk)[:original_byte_len]
                result_bytes.extend(chunk_bytes)
                chunks_processed += 1

        # Restore base LoRA state after decompression
        self._reset_lora_to_base()
        return result_bytes.hex()
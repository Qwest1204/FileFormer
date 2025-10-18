from notus.tokenizer import ByteLevelTokenizer

import hashlib
import torch


class CompressionEngine:
    def __init__(self, config : dict):
        self.config = config
        self.apply_config()

        self.tokenizer = ByteLevelTokenizer()

    def update_config(self, config):
        self.config = config
        self.apply_config()

    def apply_config(self):
        pass

    def compress(self, chunk:bytes):

        data = chunk.hex()
        hash = hashlib.sha256(chunk)

        encoded_data = self.tokenizer.encode(data)
        encoded_hash = self.tokenizer.encode(hash)

        out = {"hash": hash}
        return out

    def decompress(self, hash, bytes):
        pass

    @staticmethod
    def test_precision():
        return 0
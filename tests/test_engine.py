import unittest
import torch
from math import log2
from collections import Counter

from entropy_engine import Engine
from model import FileFormer, ByteLevelTokenizer

def shannon(boe):
    boe = Counter(boe)
    total = sum(boe.values())
    return sum(freq / total * log2(total / freq) for freq in boe.values())

def c_and_r():
    model = FileFormer(257, 256, 4, 4, 0.0)
    model.load_state_dict(
        torch.load("checkpoints/model_enwiki_pre-v0.0.1.pt",
                   map_location="cpu")['model_state_dict']
    )
    model.eval()
    engine = Engine(42, model, ByteLevelTokenizer(), 128)

    with open('data/raw/enwik512b', 'rb') as f:
        origin_data = f.read(512).hex()

    compressed_data = engine.compress(origin_data)
    recovered_data = engine.decompress(compressed_data)
    return recovered_data, origin_data, compressed_data


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.recovered_data, self.origin_data, self.compressed_data = c_and_r()

    def test_engine(self):
        self.assertEqual(self.origin_data, self.recovered_data)

    def test_entropy(self):

        self.assertLess(shannon(self.origin_data), shannon(self.compressed_data))

    def test_compression_rate(self):

        self.assertLess(len(self.compressed_data), len(self.origin_data))

if __name__ == '__main__':
    unittest.main()
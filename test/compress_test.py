import unittest

from tokenizer import Tokenizer
from engine import CompressEngine
from transformer_model import build_transformer

class MyTestCase(unittest.TestCase):
    def test_compressing(self):
        self.vocab_path = "vocab.json"
        self.merges_path = "merges.txt"

        tokenizer = Tokenizer()
        tokenizer.load_vocab_and_merges(self.vocab_path, self.merges_path)

        compress_model = build_transformer(vocab_size=5772,
                                        d_model=512,
                                        max_seq_len=256,
                                        d_ff=1024,
                                        dropout=0.1,
                                        n_layers=6,
                                        n_heads=8,
                                        factor=2,
                                        compress=True)
        correct_model = build_transformer(vocab_size=5772,
                                        d_model=512,
                                        max_seq_len=256,
                                        d_ff=1024,
                                        dropout=0.1,
                                        n_layers=6,
                                        n_heads=8,
                                        factor=16,
                                        compress=False)

        compress_engine = CompressEngine(tokenizer, compress_model, correct_model)

        with open("model_test.py", "rb") as f:
            data = f.read()

        data, mask = compress_engine.compress(data)
        print(data[0].shape)
        compress_engine.decompress(data, mask, 5)

        self.assertEqual(False, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()

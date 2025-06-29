import unittest

from notus import build_transformer
import torch

class TestModel(unittest.TestCase):
    def test_build(self):
        transformer = build_transformer(vocab_size=5666,
                                d_model=512,
                                max_seq_len=128,
                                d_ff=1024,
                                dropout=0.1,
                                n_layers=6,
                                n_heads=8,
                                factor=4,
                                compress=True,
                                winsize=64)

        self.assertIsNotNone(transformer)  # add assertion here
#TODO: update forward test for new model
    # def test_forward(self):
    #     transformer = build_transformer(vocab_size=5666,
    #                             d_model=512,
    #                             max_seq_len=128,
    #                             d_ff=1024,
    #                             dropout=0.1,
    #                             n_layers=6,
    #                             n_heads=8,
    #                             factor=4,
    #                             compress=True,
    #                             winsize=64)
    #     out_shape = transformer.forward(torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)).shape
    #     self.assertEqual(out_shape, torch.Size([1, 128, 5666]))

if __name__ == '__main__':
    unittest.main()

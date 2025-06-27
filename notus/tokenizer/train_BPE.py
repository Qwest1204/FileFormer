from BPE import BPETokenizerSimple
from pathlib import Path

files = [x for x in Path("/Users/daniilogorodnikov/dataset/app").glob('**/*') if x.is_file()]

bytes = []

for file in files:
    with open(file, 'rb') as f:
        data_from_file = f.read().hex()
        bytes.append(data_from_file)

tokenizer = BPETokenizerSimple()

tokenizer.train(bytes, 5777, {"<|endoftext|>", "<unk>"})

tokenizer.save_vocab_and_merges("vocab.json", 'merges.txt')
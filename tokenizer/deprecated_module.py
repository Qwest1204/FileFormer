def read_hex_chunks(data, chunk_size=2):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        yield chunk.hex().zfill(4)


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.special_tokens = {
            '<SOF>': 0, '<EOF>': 1, '<UNK>': 2,
            '<SOS>': 3, '<EOS>': 4, '<PAD>': 5
        }

        # Добавление hex токенов (0x0000-0xffff)
        hex_list = [f"{i:04x}" for i in range(0x10000)]
        self.vocab["enc"] = {**self.special_tokens, **{h: i + 6 for i, h in enumerate(hex_list)}}
        self.vocab["dec"] = {v: k for k, v in self.vocab["enc"].items()}

    def encode(self, bin_data):
        tokens = [self.vocab['enc']['<SOF>']]
        for chunk in read_hex_chunks(bin_data):
            tokens.append(self.vocab['enc'].get(chunk, self.vocab['enc']['<UNK>']))
        tokens.append(self.vocab['enc']['<EOF>'])
        return tokens

    def decode(self, tokens):
        return bytes.fromhex(''.join(
            self.vocab['dec'][t] for t in tokens
            if self.vocab['dec'][t] not in self.special_tokens
        ))

    def get_vocab_size(self):
        return len(self.vocab['enc'])

    def get_idx_from_token(self, token):
        return self.vocab['enc'][token]
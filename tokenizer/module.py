def read_hex_chunks(data, chunk_size=2):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        yield chunk.hex().zfill(4)  # Гарантирует 4 символа

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        enc = {'<SOF>': 0, '<EOF>': 1, '<UNK>': 2, '<SOS>': 3, '<EOS>': 4, '<PAD>': 5}
        dec = {0: '<SOF>', 1: '<EOF>', 2: '<UNK>', 3: '<SOS>', 4: '<EOS>', 5: '<PAD>'}
        hex_list = [f"{i:04x}" for i in range(0x10000)]
        for i in range(len(hex_list)):
            enc[hex_list[i]] = i + 6
            dec[i + 6] = hex_list[i]
        self.vocab["enc"] = enc
        self.vocab["dec"] = dec

    def encode(self, bin_data):
        tokens = [self.vocab['enc']['<SOF>']]
        for chunk in read_hex_chunks(bin_data):
            tokens.append(self.vocab['enc'].get(chunk, self.vocab['enc']['<UNK>']))
        tokens.append(self.vocab['enc']['<EOF>'])
        return tokens

    def decode(self, tokens):
        data = []
        for token in tokens:
            if token in [0, 1, 3, 4, 5]:
                continue  # Пропуск служебных токенов
            data.append(self.vocab['dec'][token])
        return bytes.fromhex(''.join(data))

    def get_idx_from_token(self, token):
        return self.vocab['enc'][token]
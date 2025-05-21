def read_hex_chunks(file_path, chunk_size=2, buffer_size=4096):
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                yield chunk.hex()


class Tokenizer:
    def __init__(self,):
        self.vocab = {}
        enc = {'<SOF>': 0, '<EOF>': 1, '<UNK>': 2, '<SOS>': 3, '<EOS>': 4, '<PAD>': 5}
        dec = {0: '<SOF>', 1: '<EOF>', 2: '<UNK>', 3: '<SOS>', 4: '<EOS>', 5: '<PAD>'}
        hex_list = [f"{i:04x}" for i in range(0x10000)]
        for i in range(len(hex_list)):
            enc[hex_list[i]] = i + 6
            dec[i + 6] = hex_list[i]
        self.vocab["enc"] = enc
        self.vocab["dec"] = dec

    def encode(self, file):
        tokens = []
        for chunk in read_hex_chunks(file):
            tokens.append(self.vocab['enc'][chunk])
        return tokens

    def decode(self, tokens):
        data = []
        for token in tokens:
            data.append(self.vocab['dec'][token])

    def get_idx_from_token(self, token):
        return self.vocab['enc'][token]
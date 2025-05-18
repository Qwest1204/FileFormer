import json

class Tokenizer:
    def __init__(self,):
        self.vocab = {}
        enc = {}
        dec = {}
        hex_list = [f"{i:04x}" for i in range(0x10000)]
        for i in range(len(hex_list)):
            enc[hex_list[i]] = i
            dec[i] = hex_list[i]
            self.vocab["enc"] = enc
            self.vocab["dec"] = dec

    def read_hex_chunks(self, file_path, chunk_size=2, buffer_size=4096):
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    yield chunk.hex()

    def encode(self, file):
        tokens = []
        for chunk in self.read_hex_chunks(file):
            try:
                tokens.append(self.vocab['enc'][chunk])
            except:
                tokens.append("<NON>")
        return tokens

    def decode(self, tokens):
        data = []
        for token in tokens:
            try:
                data.append(self.vocab['dec'][token])
            except:
                data.append("<NON>")
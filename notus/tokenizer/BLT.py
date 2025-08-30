class ByteLevelTokenizer:
    def __init__(self):
        # Специальные токены
        self.special_tokens = [
            "<mask>", "<pad>",
        ]

        # Создание словаря
        self.token2id = {}
        self.id2token = {}

        # Регистрация специальных токенов
        for idx, token in enumerate(self.special_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token

        # Регистрация байтов (от 00 до FF)
        for byte in range(256):
            token = format(byte, '02x')  # Байт в hex формате
            token_id = len(self.special_tokens) + byte
            self.token2id[token] = token_id
            self.id2token[token_id] = token

    def encode(self, text: str) -> list:
        if all(c in "0123456789abcdef" for c in text):  # Чистый hex
            return [self.token2id[text[i:i+2]] for i in range(0, len(text), 2)]
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for token in self.special_tokens:
                if text.startswith(token, i):
                    tokens.append(self.token2id[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                char = text[i:i+2]  # Бери сразу 2 символа для hex
                tokens.append(self.token2id.get(char, self.token2id["<unk>"]))
                i += 2
        return tokens

    def decode(self, token_ids: list) -> str:
        parts = []
        byte_buffer = bytearray()
        for token_id in token_ids:
            token = self.id2token.get(token_id, "<unk>")
            if token in self.special_tokens:
                if byte_buffer:
                    parts.append(byte_buffer.hex())
                    byte_buffer.clear()
                parts.append(token)
            else:
                byte_buffer.append(int(token, 16))
        if byte_buffer:
            parts.append(byte_buffer.hex())
        return ''.join(parts)

    @property
    def vocab_size(self) -> int:
        return len(self.special_tokens) + 256

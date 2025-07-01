class ByteLevelTokenizer:
    def __init__(self):
        # Специальные токены
        self.special_tokens = [
            "<unk>", "<sos>", "<mask>", "<pad>",
            "<eos>", "<eof>", "<sof>", "<t1>",
            "<t2>", "<t3>", "<t4>"
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
        tokens = []
        i = 0
        n = len(text)

        while i < n:
            matched = False

            # Проверка специальных токенов
            for token in self.special_tokens:
                if text.startswith(token, i):
                    tokens.append(self.token2id[token])
                    i += len(token)
                    matched = True
                    break

            # Обработка обычных символов
            if not matched:
                char = text[i]
                byte_repr = char.encode('utf-8')
                for b in byte_repr:
                    hex_byte = format(b, '02x')
                    tokens.append(self.token2id[hex_byte])
                i += 1

        return tokens

    def decode(self, token_ids: list) -> str:
        parts = []
        byte_buffer = bytearray()

        for token_id in token_ids:
            token = self.id2token.get(token_id, "<unk>")

            if token in self.special_tokens:
                # Декодируем накопленные байты
                if byte_buffer:
                    parts.append(byte_buffer.decode('utf-8'))
                    byte_buffer.clear()
                parts.append(token)
            else:
                # Добавляем байт в буфер
                byte_buffer.append(int(token, 16))

        # Декодируем оставшиеся байты
        if byte_buffer:
            parts.append(byte_buffer.decode('utf-8'))

        return ''.join(parts)

    @property
    def vocab_size(self) -> int:
        return len(self.special_tokens) + 256


# Пример использования
if __name__ == "__main__":
    tokenizer = ByteLevelTokenizer()

    # Тест кодирования
    text = "ffe45a7e8ccc"
    encoded = tokenizer.encode(text)
    print("Закодированный текст:", encoded)

    # Тест декодирования
    decoded = tokenizer.decode(encoded)
    print("Декодированный текст:", decoded)

    # Проверка словаря
    print("\nПримеры токенов:")
    print("<sos> =", tokenizer.token2id["<sos>"])
    print("00 =", tokenizer.token2id["00"])
    print("ff =", tokenizer.token2id["ff"])
    print("Размер словаря:", tokenizer.vocab_size)
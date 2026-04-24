class ByteLevelTokenizer:
    def __init__(self):
        # Специальные токены (добавлен <UNK>)
        self.special_tokens = [
            "<PAD>", "<MASK>", "<BOS>", "<SOS>", "<EOS>", "<UNK>"
        ]

        self.token2id = {}
        self.id2token = {}

        # Регистрация специальных токенов
        for idx, token in enumerate(self.special_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token

        # Одиночные байты (00..FF)
        for byte in range(256):
            token = format(byte, '02x')          # всегда 2 символа
            token_id = len(self.special_tokens) + byte
            self.token2id[token] = token_id
            self.id2token[token_id] = token

        # Пары байтов (0000..FFFF) – двухбайтовые токены
        for pair in range(256 * 256):
            token = format(pair, '04x')          # всегда 4 символа
            token_id = len(self.special_tokens) + 256 + pair
            self.token2id[token] = token_id
            self.id2token[token_id] = token

    def encode(self, text: str) -> list:
        """
        Кодирует входную строку.
        - Если строка состоит только из hex-символов и имеет чётную длину,
          считается, что это уже байтовое представление в hex.
        - Иначе текст преобразуется в UTF-8 байты, а затем в hex.
        Затем жадным алгоритмом выбираются токены:
          сначала пытаемся взять 4 символа (2 байта), если не помещается — 2 символа.
        """
        # Определяем, является ли вход hex-строкой
        hex_chars = set("0123456789abcdef")
        if all(c in hex_chars for c in text) and len(text) % 2 == 0:
            hex_str = text
        else:
            # Преобразуем обычный текст в байты UTF-8, затем в hex
            byte_repr = text.encode('utf-8')
            hex_str = byte_repr.hex()

        tokens = []
        i = 0
        n = len(hex_str)

        # Жадное кодирование: сначала двухбайтовые токены, потом однобайтовый остаток
        while i < n:
            if i + 4 <= n:
                # Пытаемся взять 4 символа (2 байта)
                token4 = hex_str[i:i+4]
                tokens.append(self.token2id[token4])
                i += 4
            else:
                # Остался 1 байт (2 символа)
                token2 = hex_str[i:i+2]
                tokens.append(self.token2id[token2])
                i += 2

        return tokens

    def decode(self, token_ids: list) -> str:
        """
        Декодирует последовательность ID обратно в строку.
        Накопленные байты выводятся как hex, специальные токены — как есть.
        """
        parts = []
        byte_buffer = bytearray()

        for token_id in token_ids:
            token = self.id2token.get(token_id, "<UNK>")
            if token in self.special_tokens:
                # Сливаем накопленные байты перед специальным токеном
                if byte_buffer:
                    parts.append(byte_buffer.hex())
                    byte_buffer.clear()
                parts.append(token)
            else:
                # Токен содержит hex представление 1 или 2 байт
                value = int(token, 16)
                if len(token) == 2:      # одиночный байт
                    byte_buffer.append(value)
                elif len(token) == 4:    # два байта
                    byte_buffer.extend(value.to_bytes(2, 'big'))
                # Игнорируем другие длины (по логике невозможны)

        if byte_buffer:
            parts.append(byte_buffer.hex())

        return ''.join(parts)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)
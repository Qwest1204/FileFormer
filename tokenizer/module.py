def read_hex_chunks(data, chunk_size=2):
    """Generator function to split binary data into fixed-size hex string chunks.

    Iterates over binary data in specified chunk sizes, converts each chunk to a hexadecimal string,
    and zero-pads it to ensure a consistent 4-character length per chunk.

    Args:
        data (bytes): Binary input data to be processed.
        chunk_size (int): Number of bytes per chunk. Default is 2 (4 hex characters).

    Yields:
        str: 4-character hexadecimal string representation of each data chunk.
            Example: '01ab' for a 2-byte chunk.
    """
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        yield chunk.hex().zfill(4)  # Ensures 4 symbols using zero-padding


class Tokenizer:
    """Converts between binary data and tokenized integer sequences using a hex-based vocabulary.

    Maintains:
    - enc: Token-to-index mapping (str -> int)
    - dec: Index-to-token mapping (int -> str)
    Vocabulary includes special control tokens and all possible 4-character hex values.
    """

    def __init__(self):
        """Initializes vocabulary with:
        - 6 special control tokens (<SOF>, <EOF>, etc.)
        - 65,536 hex tokens (0000-ffff) representing all possible 2-byte combinations
        """
        self.vocab = {}
        # Special tokens dictionary (control characters)
        enc = {'<SOF>': 0, '<EOF>': 1, '<UNK>': 2, '<SOS>': 3, '<EOS>': 4, '<PAD>': 5}
        dec = {0: '<SOF>', 1: '<EOF>', 2: '<UNK>', 3: '<SOS>', 4: '<EOS>', 5: '<PAD>'}

        # Generate all 4-character hex strings (0000-ffff)
        hex_list = [f"{i:04x}" for i in range(0x10000)]

        # Add hex tokens to vocabulary starting from index 6
        for i in range(len(hex_list)):
            enc[hex_list[i]] = i + 6
            dec[i + 6] = hex_list[i]

        self.vocab["enc"] = enc
        self.vocab["dec"] = dec

    def encode(self, bin_data):
        """Converts binary data to a sequence of token indices.

        Process:
        1. Adds Start-of-File (<SOF>) token
        2. Converts every 2-byte chunk to corresponding hex token
        3. Uses <UNK> token for invalid hex chunks (though this should never occur)
        4. Appends End-of-File (<EOF>) token

        Args:
            bin_data (bytes): Input binary data to encode

        Returns:
            list[int]: Token indices sequence including control tokens
        """
        tokens = [self.vocab['enc']['<SOF>']]  # Start with SOF
        for chunk in read_hex_chunks(bin_data):
            # Get token ID or fallback to <UNK> (unlikely with proper 2-byte chunks)
            tokens.append(self.vocab['enc'].get(chunk, self.vocab['enc']['<UNK>']))
        tokens.append(self.vocab['enc']['<EOF>'])  # End with EOF
        return tokens

    def decode(self, tokens):
        """Converts token indices back to original binary data.

        Process:
        1. Filters out control tokens (SOF/EOF/SOS/EOS/PAD)
        2. Converts remaining tokens to hex strings
        3. Joins hex strings and converts to bytes

        Args:
            tokens (list[int]): Sequence of token indices to decode

        Returns:
            bytes: Reconstructed binary data

        Note:
            Will raise KeyError if encountering invalid token indices
        """
        data = []
        for token in tokens:
            # Skip control tokens (only process hex tokens)
            if token in [0, 1, 3, 4, 5]:
                continue
            data.append(self.vocab['dec'][token])
        return bytes.fromhex(''.join(data))  # Convert hex string to bytes

    def get_idx_from_token(self, token):
        """Utility method to get index for a given token string.

        Args:
            token (str): Token string to look up (either control token or 4-char hex)

        Returns:
            int: Corresponding token index

        Raises:
            KeyError: If token is not in vocabulary
        """
        return self.vocab['enc'][token]  # Direct dictionary lookup
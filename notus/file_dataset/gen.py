import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm
from notus import ByteLevelTokenizer

# Constants
MAX_SEQ_LENGTH = 8192
tokenizer = ByteLevelTokenizer()
pad_token = tokenizer.encode("<pad>")[0]

# Argument parser setup
parser = argparse.ArgumentParser(description='Dataset generator')
parser.add_argument(
    '--data',
    type=str,
    default="/",
    help="Path to input files for dataset generation"
)
args = parser.parse_args()

def read_and_tokenize(file: Path) -> list:
    """
    Read a file in binary mode and tokenize its hex representation.

    Args:
        file: Path object representing the input file
    Returns:
        List of tokens from the hex-encoded file content
    """
    all_tokens = []
    with open(file, 'rb') as f:
        while True:
            byte_chunk = f.read(4096)
            if not byte_chunk:
                break
            hex_chunk = byte_chunk.hex()
            tokens = tokenizer.encode(hex_chunk)
            all_tokens.extend(tokens)
    return all_tokens

def gen_dataset():
    """
    Generate dataset by processing files, tokenizing them, and saving as PyTorch tensors.
    Each tensor is saved as Origin_data_{index}.pt in the output directory.
    """
    # Get all files recursively from the input directory
    files = [x for x in Path(args.data).glob('**/*') if x.is_file()]

    # Create output directory
    output_dir = "origin_data"
    os.makedirs(output_dir, exist_ok=True)

    index = 0
    for file in tqdm(files, desc="Processing files"):
        try:
            # Read and tokenize file content
            all_tokens = read_and_tokenize(file)

            # Process tokens in chunks of MAX_SEQ_LENGTH
            for i in range(0, len(all_tokens), MAX_SEQ_LENGTH):
                chunk = all_tokens[i:i + MAX_SEQ_LENGTH]

                # Pad chunk if necessary
                if len(chunk) < MAX_SEQ_LENGTH:
                    chunk += [pad_token] * (MAX_SEQ_LENGTH - len(chunk))

                # Convert to tensor and save
                tensor = torch.tensor(chunk, dtype=torch.long)
                output_path = os.path.join(output_dir, f"Origin_data_{index}.pt")
                torch.save(tensor, output_path)
                index += 1

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

if __name__ == "__main__":
    gen_dataset()
import torch
from notus import ByteLevelTokenizer

def evaluate(forward, x):
    tokens, masked_tokens, pads, hash, extention_tokenize = x
    tokenizer = ByteLevelTokenizer()
    # Получаем mask_token и pad_token через tokenizer
    mask_token = tokenizer.encode("<mask>")[0]
    pad_token = tokenizer.encode("<pad>")[0]

    with torch.no_grad():
        decoder_out = forward(masked_tokens, hash, extention_tokenize, pads)
        # Получаем предсказанные токены
        predicted_tokens = torch.argmax(decoder_out, dim=2)  # (bs, seq_len)

        # Вывод первых 40 токенов для первого элемента батча
        print(f"origin tokens : {tokenizer.decode(tokens[0].detach().cpu().tolist()[:40])}")
        print(f"gen tokens : {tokenizer.decode(predicted_tokens[0].detach().cpu().tolist()[:40])}")

        # Вычисление точности для маскированных токенов
        mask_positions = masked_tokens == mask_token  # (bs, seq_len)
        if mask_positions.any():
            masked_predictions = predicted_tokens[mask_positions]  # Предсказания для <mask>
            masked_targets = tokens[mask_positions]  # Истинные токены для <mask>
            masked_correct = (masked_predictions == masked_targets).float().sum()
            masked_accuracy = (masked_correct / mask_positions.sum()).item() * 100
        else:
            masked_accuracy = 0.0
            print("No masked tokens found in the batch.")

        # Вычисление точности для всех токенов (исключая <pad>)
        non_pad_positions = tokens != pad_token  # (bs, seq_len)
        if non_pad_positions.any():
            non_pad_predictions = predicted_tokens[non_pad_positions]
            non_pad_targets = tokens[non_pad_positions]
            total_correct = (non_pad_predictions == non_pad_targets).float().sum()
            total_accuracy = (total_correct / non_pad_positions.sum()).item() * 100
        else:
            total_accuracy = 0.0
            print("No non-pad tokens found in the batch.")

        # Вывод точности
        print(f"Masked tokens accuracy: {masked_accuracy:.2f}%")
        print(f"Total tokens accuracy (excluding padding): {total_accuracy:.2f}%")
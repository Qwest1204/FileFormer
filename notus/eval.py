import torch

def evaluate(encoder, decoder, tokenizer, x):
    tokens, masked_tokens, pads, hash, extention_tokenize = x
    with torch.no_grad():
        encoder_out = encoder(hash, extention_tokenize)
        decoder_out = decoder(masked_tokens, encoder_out, pads)

        print(f"origin tokens : {tokenizer.decode(tokens[0].detach().cpu().tolist()[:40])}")

        out = torch.argmax(decoder_out, dim=2)[0].detach().cpu().tolist()

        print(f"gen tokens : {tokenizer.decode(out[:40])}")
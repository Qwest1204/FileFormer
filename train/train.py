from notus import Encoder, Decoder, utils, FileDataset, Muon
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

#configs
configs = utils.load_config("../config/config.yml")
device = configs['train']["device"]
print("configs successfully loaded, used device: ", device)

#dataset
dataset = FileDataset(**configs['dataset'])
dataset.prepare()
dataset_len = len(dataset)
dataloader = DataLoader(dataset, batch_size=configs['train']['batch_size'],
                        shuffle=True, num_workers=configs['train']['num_workers']
                        )

print("dataloader successfully loaded, used data: ", configs['dataset']['path'])

#models
encoder = Encoder(**configs['encoder']).to(device)
decoder = Decoder(**configs['decoder']).to(device)
print("encoder successfully loaded, num paramns: ", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
print("decoder successfully loaded, num paramns: ", sum(p.numel() for p in decoder.parameters() if p.requires_grad))
#optims
optimizer = Muon(iter(list(encoder.parameters()) + list(decoder.parameters())), lr=configs['train']['lr'])

#loss
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

def train():
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(dataloader):
        tokens, masked_tokens, pads, hash, extention_tokenize = data

        tokens = tokens.to(device)
        masked_tokens = masked_tokens.to(device)
        pads = pads.to(device)
        hash = hash.to(device)
        extention_tokenize = extention_tokenize.to(device)

        optimizer.zero_grad()

        encoder_out = encoder(hash, extention_tokenize)

        decoder_out = decoder(masked_tokens, encoder_out, pads)

        loss = loss_fn(decoder_out.view(-1, configs['encoder']['vocab_size']), tokens.view(-1))
        loss.backward()
        print(loss.item())
        optimizer.step()
        running_loss += loss.item()
        last_loss = running_loss/dataset_len
        if i % 1000 == 999:
            print("total loss: {:.4f}".format(last_loss))
    return last_loss

def main():
    epoch_number = 0
    for epoch in range(configs['train']['num_epoch']):
        print("epoch: ", epoch+1)

        encoder.train()
        decoder.train()

        avg_loss = train()

        print("avg_loss: ", avg_loss)

        utils.save_model(encoder, optimizer, configs['encoder'], f"encoder_{epoch_number}.pth")
        utils.save_model(decoder, optimizer, configs['decoder'], f"decoder{epoch_number}.pth")

        epoch_number += 1

main()





from notus import Encoder, Decoder, utils, FileDataset, Muon
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

#configs
configs = utils.load_config("/config/config.yml")
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
optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=configs['train']['lr'])

#loss
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

def train_epoch(encoder, decoder, dataloader, optimizer, loss_fn, device, configs, dataset_len):
    """
    Trains the model for one epoch.

    Args:
        encoder: The encoder model.
        decoder: The decoder model.
        dataloader: DataLoader for the training data.
        optimizer: Optimizer for the models.
        loss_fn: Loss function.
        device: Device to run on (e.g., 'cuda' or 'cpu').
        configs: Configuration dictionary.
        dataset_len: Length of the dataset (for average loss calculation).

    Returns:
        Average loss for the epoch.
    """
    encoder.train()
    decoder.train()

    running_loss = 0.0
    num_batches = len(dataloader)  # For potential use if dataset_len is num_batches

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

        # Optional: Print loss per batch (comment out if too verbose)
        # print(f"Batch {i+1} loss: {loss.item():.4f}")

        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            current_avg = running_loss / (i + 1)
            print(f"Batch {i+1} - Current average loss: {current_avg:.4f}")

    avg_loss = running_loss / dataset_len  # Or use / num_batches if appropriate
    print(f"Epoch total average loss: {avg_loss:.4f}")
    return avg_loss

def main(encoder, decoder, dataloader, optimizer, loss_fn, device, configs, utils, dataset_len):
    """
    Main training loop.

    Args:
        encoder: The encoder model.
        decoder: The decoder model.
        dataloader: DataLoader for the training data.
        optimizer: Optimizer for the models.
        loss_fn: Loss function.
        device: Device to run on.
        configs: Configuration dictionary.
        utils: Utility module for saving models.
        dataset_len: Length of the dataset.
    """
    num_epochs = configs['train']['num_epoch']

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")

        avg_loss = train_epoch(encoder, decoder, dataloader, optimizer, loss_fn, device, configs, dataset_len)

        print(f"Average loss for epoch {epoch + 1}: {avg_loss:.4f}")

        # Save models
        utils.save_model(encoder, optimizer, configs['encoder'], f"encoder_{epoch}.pth")
        utils.save_model(decoder, optimizer, configs['decoder'], f"decoder_{epoch}.pth")

main()





import torch
import torch.nn as nn
import torch.optim as optim
import os

from model import Encoder, Decoder, Seq2Seq

# Set random seeds for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


# Assuming model, optimizer, and criterion setup as before

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        # Reshape output for loss calculation
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def save_model(model, path):
    torch.save(model.state_dict(), path)


# Training loop with model saving
N_EPOCHS = 10
CLIP = 1
BEST_LOSS = float('inf')
MODEL_DIR = 'saved_models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

for epoch in range(N_EPOCHS):

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')

    # Save model after each epoch
    epoch_save_path = os.path.join(MODEL_DIR, f'model_epoch_{epoch + 1}.pt')
    save_model(model, epoch_save_path)
    print(f'Model saved to {epoch_save_path}')

    # Check if this is the best model so far
    if train_loss < BEST_LOSS:
        BEST_LOSS = train_loss
        best_model_path = os.path.join(MODEL_DIR, 'best_model.pt')
        save_model(model, best_model_path)
        print(f'New best model saved to {best_model_path}')

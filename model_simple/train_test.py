import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# import datasets and transforms
from datasets import SpeechDataset
from transforms import *
from utils import collate_fn

# define hyperparameters
batch_size = 32
lr = 0.001
num_epochs = 10
input_dim = 161 # number of Mel filterbank channels
output_dim = 4000 # number of Chinese characters
d_model = 256
n_head = 4
dim_feedforward = 1024
num_layers = 6
dropout_rate = 0.1

# load datasets and apply transforms
train_dataset = SpeechDataset('train.csv', Compose([LoadAudio(), ToMelSpectrogram(), Normalize(), ToTensor()]))
val_dataset = SpeechDataset('val.csv', Compose([LoadAudio(), ToMelSpectrogram(), Normalize(), ToTensor()]))

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# create model and optimizer
model = Conformer(input_dim, output_dim, d_model, n_head, dim_feedforward, num_layers, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=lr)

# define loss function
criterion = nn.CTCLoss(blank=output_dim-1)

# train and evaluate model
for epoch in range(num_epochs):
    # train
    model.train()
    for x, y, x_lengths, y_lengths in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        y_pred_lengths = torch.full(size=(y_pred.size(0),), fill_value=y_pred.size(1), dtype=torch.int32)
        loss = criterion(y_pred.transpose(0, 1), y, y_pred_lengths, y_lengths)
        loss.backward()
        optimizer.step()
    print('Epoch {}/{} - train loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # evaluate
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for x, y, x_lengths, y_lengths in val_loader:
            y_pred = model(x)
            y_pred_lengths = torch.full(size=(y_pred.size(0),), fill_value=y_pred.size(1), dtype=torch.int32)
            loss = criterion(y_pred.transpose(0, 1), y, y_pred_lengths, y_lengths)
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(val_dataset)
        print('Epoch {}/{} - val loss: {:.4f}'.format(epoch+1, num_epochs, avg_loss))
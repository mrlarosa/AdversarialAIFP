#from os import pread
import torch
import torch.nn as nn
from scipy.io.wavfile import read
from torch.utils.data import Dataset

import numpy as np


class sound_data(Dataset):

    def __init__(self, real_paths, fake_paths, transform=None):
        real_data = [read(fp)[1] for fp in real_paths]
        fake_data = [read(fp)[1] for fp in fake_paths]
        real_labels = np.zeros((len(real_paths),1), dtype=np.float32)
        fake_labels = np.ones((len(fake_paths),1), dtype=np.float32)
        self.data = real_data + fake_data
        self.labels = np.concatenate((real_labels, fake_labels))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            item = self.transform(item)
        return [item, label]

class LSTMmodel(nn.Module):
    def __init__(self, input_size, latent_size, num_layers=2):
        super(LSTMmodel, self).__init__()

        self.latent_size = latent_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.lstm_enc = nn.LSTM(input_size = input_size, hidden_size = latent_size, num_layers = num_layers)
        self.linear = nn.Sequential(nn.Linear(latent_size, latent_size),
                                    nn.Tanh(),
                                    nn.Linear(latent_size, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        z, _ = self.lstm_enc(x)
        pred = self.linear(z)
        return pred
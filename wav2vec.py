import torch
import torch.nn as nn
from scipy.io.wavfile import read
from torch.utils.data import Dataset

import numpy as np

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
import torchaudio



class Wav2Vec2_Data(Dataset):
    def __init__(self, real_paths, fake_paths, transform=None):
        real_data = [read(fp)[1] for fp in real_paths]
        fake_data = [read(fp)[1] for fp in fake_paths]
        real_labels = np.zeros((len(real_paths),1), dtype=np.float32)
        fake_labels = np.ones((len(fake_paths),1), dtype=np.float32)
        self.data = real_data + fake_data
        self.labels = np.concatenate((real_labels, fake_labels))
        self.transform = transform
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        #waveform, sample_rate = torchaudio.load(self.paths[idx])

        #resample to 16 kHz
        #if sample_rate != 16000:
            #resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            #waveform = resampler(waveform)

        #remove channel dimension
        #waveform = item.squeeze(0)  # Shape: (time,)
        waveform = item

        #preprocess with Wav2Vec2Processor
        input_values = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_values.squeeze(0)  # Shape: (seq_len,)


        if self.transform:
            input_values = self.transform(input_values)
        return [input_values, label]
    

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, wav2vec_model_name, mlp_hidden_dim1, mlp_hidden_dim2, num_classes):
        super(Wav2Vec2Classifier, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        self.mlp = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, mlp_hidden_dim1),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim1, mlp_hidden_dim2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim2, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, input_values):
        wav2vec_output = self.wav2vec2(input_values).last_hidden_state  #shape: (batch_size, seq_len, hidden_size)
        
        #pool features (mean pooling over the sequence length)
        features = wav2vec_output.mean(dim=1)  #shape: (batch_size, hidden_size)
        
        y = self.mlp(features)  # Shape: (batch_size, num_classes)
        return y




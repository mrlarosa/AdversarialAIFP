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
        try:
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
        except Exception as e:
            print("Eception Occured")
        return [input_values, label]
    

import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, wav2vec_model_name, mlp_hidden_dim1, mlp_hidden_dim2, num_classes, train_wav2vec=False):
        super(Wav2Vec2Classifier, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        
        # Freeze Wav2Vec2 parameters unless train_wav2vec is True
        for param in self.wav2vec2.parameters():
            param.requires_grad = train_wav2vec
        
        # Initialize the MLP layers
        self.linear1 = nn.Linear(self.wav2vec2.config.hidden_size, mlp_hidden_dim1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(mlp_hidden_dim1, mlp_hidden_dim2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(mlp_hidden_dim2, mlp_hidden_dim2//2)
        self.sigmoid1 = nn.ReLU()
        self.linear4 = nn.Linear(mlp_hidden_dim2//2, num_classes)
        self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, input_values):
        # Get Wav2Vec2 output
        wav2vec_output = self.wav2vec2(input_values).last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        
        # Pool features (mean pooling over the sequence length)
        features = wav2vec_output.mean(dim=1)  # Shape: (batch_size, hidden_size)
        
        # Pass through the MLP
        x = self.linear1(features)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid1(x)
        x = self.linear4(x)
        y = self.sigmoid2(x)
        return y

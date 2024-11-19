import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from soundmodel import sound_data, LSTMmodel
import torch.optim as optim
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

from wav2vec import Wav2Vec2_Data, Wav2Vec2Classifier

bs= 512
NUM_EPOCHS = 100
criterion = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "C:/Users/razaa/Downloads/AdversarialAIFP/for-2seconds/"
training_real = root + "training/real"
training_fake = root + "validation/fake"
validation_real = root + "training/real"
validation_fake = root + "validation/fake"

training_real = [training_real + '/'+ f for f in listdir(training_real) if isfile(join(training_real, f))]
training_fake = [training_fake + '/'+ f for f in listdir(training_fake) if isfile(join(training_fake, f))]

validation_real = [validation_real + '/'+ f for f in listdir(validation_real) if isfile(join(validation_real, f))]
validation_fake = [validation_fake + '/'+ f for f in listdir(validation_fake) if isfile(join(validation_fake, f))]

print("W")

tr_ds = Wav2Vec2_Data(training_real, training_fake)
val_ds = Wav2Vec2_Data(validation_real, validation_fake)

print("L")

tr_ldr = DataLoader(tr_ds, batch_size=bs, shuffle=True)
val_ldr = DataLoader(val_ds, batch_size=bs, shuffle=True)



wav2vec_model_name = "facebook/wav2vec2-base"
model = Wav2Vec2Classifier(wav2vec_model_name, 64, 16, 1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

###############
# Train Model #
###############
tr_losses = []
val_losses = []
for epoch in tqdm(range(NUM_EPOCHS)):
    tr_loss = 0
    val_loss = 0

    model.train()
    for data, label in tr_ldr:
        data = data.to(device)
        label = label.to(device)
        
        pred = model(data)
        loss = criterion(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for data, label in val_ldr:
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)
            val_loss += loss.item()

    print("Training: ", tr_loss, "\tValidation: ", val_loss)
    tr_losses.append(tr_loss / len(tr_ldr))
    val_losses.append(val_loss / len(val_ldr))
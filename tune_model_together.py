import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from wav2vec import Wav2Vec2_Data, Wav2Vec2Classifier
import torch.optim as optim
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pickle

bs= 64
NUM_EPOCHS = 100
criterion = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp = "combined_finetuned_model"

#######
# Normal Dataset
#######
datasets = ["Normal", "Combinded"]
ldr = {}

#Normal Data
root = "/scratch/mrl78/AdversarialAIFP/Data/Normal/"
training_real = root + "training/real"
training_fake = root + "training/fake"
validation_real = root + "validation/real"
validation_fake = root + "validation/fake"
testing_real = root + "testing/real"
testing_fake = root + "testing/fake"
training_real = [training_real + '/'+ f for f in listdir(training_real) if isfile(join(training_real, f))]
training_fake = [training_fake + '/'+ f for f in listdir(training_fake) if isfile(join(training_fake, f))]
validation_real = [validation_real + '/'+ f for f in listdir(validation_real) if isfile(join(validation_real, f))]
validation_fake = [validation_fake + '/'+ f for f in listdir(validation_fake) if isfile(join(validation_fake, f))]
testing_real = [testing_real + '/'+ f for f in listdir(testing_real) if isfile(join(testing_real, f))]
testing_fake = [testing_fake + '/'+ f for f in listdir(testing_fake) if isfile(join(testing_fake, f))]
tr_ds = Wav2Vec2_Data(training_real, training_fake)
val_ds = Wav2Vec2_Data(validation_real, validation_fake)
te_ds = Wav2Vec2_Data(testing_real, testing_fake)
tr_ldr = DataLoader(tr_ds, batch_size=bs, shuffle=True)
val_ldr = DataLoader(val_ds, batch_size=bs, shuffle=True)
te_ldr = DataLoader(te_ds, batch_size=bs, shuffle=True)
d_ldr = {"train":tr_ldr, "validation":val_ldr, "test":te_ldr}
ldr["Normal"] = d_ldr

#Children Files
root = "/scratch/mrl78/AdversarialAIFP/Data/Children/"
c_training_real = root + "training/real"
c_training_fake = root + "training/fake"
c_validation_real = root + "validation/real"
c_validation_fake = root + "validation/fake"
c_testing_real = root + "testing/real"
c_testing_fake = root + "testing/fake"
c_training_real = [c_training_real + '/'+ f for f in listdir(c_training_real) if isfile(join(c_training_real, f))]
c_training_fake = [c_training_fake + '/'+ f for f in listdir(c_training_fake) if isfile(join(c_training_fake, f))]
c_validation_real = [c_validation_real + '/'+ f for f in listdir(c_validation_real) if isfile(join(c_validation_real, f))]
c_validation_fake = [c_validation_fake + '/'+ f for f in listdir(c_validation_fake) if isfile(join(c_validation_fake, f))]
c_testing_real = [c_testing_real + '/'+ f for f in listdir(c_testing_real) if isfile(join(c_testing_real, f))]
c_testing_fake = [c_testing_fake + '/'+ f for f in listdir(c_testing_fake) if isfile(join(c_testing_fake, f))]

#Stuttering Files
root = "/scratch/mrl78/AdversarialAIFP/Data/Children/"
s_training_real = root + "training/real"
s_training_fake = root + "training/fake"
s_validation_real = root + "validation/real"
s_validation_fake = root + "validation/fake"
s_testing_real = root + "testing/real"
s_testing_fake = root + "testing/fake"
s_training_real = [s_training_real + '/'+ f for f in listdir(s_training_real) if isfile(join(s_training_real, f))]
s_training_fake = [s_training_fake + '/'+ f for f in listdir(s_training_fake) if isfile(join(s_training_fake, f))]
s_validation_real = [s_validation_real + '/'+ f for f in listdir(s_validation_real) if isfile(join(s_validation_real, f))]
s_validation_fake = [s_validation_fake + '/'+ f for f in listdir(s_validation_fake) if isfile(join(s_validation_fake, f))]
s_testing_real = [s_testing_real + '/'+ f for f in listdir(s_testing_real) if isfile(join(s_testing_real, f))]
s_testing_fake = [s_testing_fake + '/'+ f for f in listdir(s_testing_fake) if isfile(join(s_testing_fake, f))]

#Combine Files
combo_training_real = s_training_real + c_training_real
combo_training_fake = s_training_fake + c_training_fake
combo_validation_real = s_validation_real + c_validation_real
combo_validation_fake = s_validation_fake + c_validation_fake
combo_testing_real = s_testing_real + c_testing_real
combo_testing_fake = s_testing_fake + c_testing_fake

tr_ds = Wav2Vec2_Data(combo_training_real, combo_training_fake)
val_ds = Wav2Vec2_Data(combo_validation_real, combo_validation_fake)
te_ds = Wav2Vec2_Data(combo_testing_real, combo_testing_fake)
tr_ldr = DataLoader(tr_ds, batch_size=bs, shuffle=True)
val_ldr = DataLoader(val_ds, batch_size=bs, shuffle=True)
te_ldr = DataLoader(te_ds, batch_size=bs, shuffle=True)
d_ldr = {"train":tr_ldr, "validation":val_ldr, "test":te_ldr}
ldr["Combo"] = d_ldr

#######
# Model Params
#######

wav2vec_model_name = "facebook/wav2vec2-base"
model = Wav2Vec2Classifier(wav2vec_model_name, 64, 16, 1, False)
model_save_path = f"/scratch/mrl78/AdversarialAIFP/saves/{exp}/model_just_mlp.pt"

#Fine tune last 2 layers
for param in model.linear3.parameters():
    param.requires_grad = False
for param in model.linear4.parameters():
    param.requires_grad = False

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.to(device)
model.load_state_dict(torch.load("/scratch/mrl78/AdversarialAIFP/saves/basic_model/model_just_mlp.pt"))

optimizer = optim.AdamW(model.parameters(), lr=0.00001)

###############
# Train Model #
###############
tr_losses = []
losses_normal = []
losses_combo = []

for epoch in tqdm(range(NUM_EPOCHS)):
    tr_loss = 0
    val_loss = 0
    combo_loss = 0

    model.train()
    for data, label in ldr["Combo"]["train"]:
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
        for data, label in ldr["Normal"]["validation"]:
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)
            val_loss += loss.item()

    with torch.no_grad():
        for data, label in ldr["Combo"]["validation"]:
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)
            combo_loss += loss.item()
 
    print("Training: ", tr_loss, "\tValidation: ", val_loss)
    tr_losses.append(tr_loss / len(ldr["Combo"]["train"]))
    losses_normal.append(val_loss / len(ldr["Normal"]["validation"]))
    losses_combo.append(combo_loss / len(ldr["Combo"]["validation"]))

torch.save(model.state_dict(), model_save_path)

# Save lists
with open(f"/scratch/mrl78/AdversarialAIFP/saves/{exp}/losses.pkl", "wb") as f:
    pickle.dump({"tr_losses": tr_losses,
                 "losses_normal": losses_normal, 
                 "losses_combo": losses_combo}, f)

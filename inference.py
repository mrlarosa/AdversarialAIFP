import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from os import listdir
from os.path import isfile, join

from wav2vec import Wav2Vec2_Data, Wav2Vec2Classifier

def infer(data_root, model_save_path):
    bs= 100
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_real = data_root + "/real"
    validation_fake = data_root + "/fake"

    validation_real = [validation_real + '/'+ f for f in listdir(validation_real) if isfile(join(validation_real, f))]
    validation_fake = [validation_fake + '/'+ f for f in listdir(validation_fake) if isfile(join(validation_fake, f))]

    val_ds = Wav2Vec2_Data(validation_real, validation_fake)

    val_ldr = DataLoader(val_ds, batch_size=bs, shuffle=True)

    wav2vec_model_name = "facebook/wav2vec2-base"
    model = Wav2Vec2Classifier(wav2vec_model_name, 64, 16, 1, False)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    model.load_state_dict(torch.load(model_save_path))

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data, label in val_ldr:
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)
            val_loss += loss.item()

    return val_loss
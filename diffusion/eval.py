import torch
from torch.utils import data

torch.set_grad_enabled(False)
from training_mse import Network
import soundfile as sf

from pytorch_lightning.callbacks import ModelCheckpoint
from random import randint

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from diffusion_dataset import DiffusionDataset
from transforms import PitchTransformer, LoudnessTransformer
import numpy as np
from tqdm import tqdm

import pickle

list_transforms = [
    (PitchTransformer, {}),
    (LoudnessTransformer, {}),
]

inst = "violin"
dataset = DiffusionDataset(instrument=inst,
                           data_augmentation=False,
                           type_set="valid",
                           list_transforms=list_transforms,
                           eval=True)

model = Network.load_from_checkpoint(
    "logs/diffusion/violin/default/version_10/checkpoints/epoch=20738-step=373301.ckpt",
    strict=False).eval()

model.set_noise_schedule()
model.ddsp = torch.jit.load("ddsp_violin_pretrained.ts").eval()

# Initialize data :

sample = np.empty(0)
time = np.empty(0)

# samples data

u_f0 = np.empty(0)
u_lo = np.empty(0)
e_f0 = np.empty(0)
e_lo = np.empty(0)
pred_f0 = np.empty(0)
pred_lo = np.empty(0)
onsets = np.empty(0)
offsets = np.empty(0)

# Prediction loops :

N_EXAMPLE = 20
for i in tqdm(range(N_EXAMPLE)):
    target, midi, ons, offs = dataset[i]

    n_step = 10
    out = model.sample(midi.unsqueeze(0), midi.unsqueeze(0))

    f0, lo = dataset.inverse_transform(out)
    midi_f0, midi_lo = dataset.inverse_transform(midi)
    target_f0, target_lo = dataset.inverse_transform(target)

    # sample information

    sample_idx = np.ones_like(f0) * i
    t = np.arange(len(f0)) / 100  # sr = 100

    sample = np.concatenate((sample, sample_idx))
    time = np.concatenate((time, t))

    # add to results:

    u_f0 = np.concatenate((u_f0, midi_f0.squeeze()))
    u_lo = np.concatenate((u_lo, midi_lo.squeeze()))

    e_f0 = np.concatenate((e_f0, target_f0.squeeze()))
    e_lo = np.concatenate((e_lo, target_lo.squeeze()))

    pred_f0 = np.concatenate((pred_f0, f0.squeeze()))
    pred_lo = np.concatenate((pred_lo, lo.squeeze()))

    onsets = np.concatenate((onsets, ons.squeeze()))
    offsets = np.concatenate((offsets, offs.squeeze()))

out = {
    "sample": sample,
    "time": time,
    "u_f0": u_f0,
    "u_lo": u_lo,
    "e_f0": e_f0,
    "e_lo": e_lo,
    "pred_f0": pred_f0,
    "pred_lo": pred_lo,
    "onsets": onsets,
    "offsets": offsets
}

with open("results/diffusion/data/{}-results.pickle".format(inst),
          "wb") as file_out:
    pickle.dump(out, file_out)

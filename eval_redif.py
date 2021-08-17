import torch
import torch.nn as nn

torch.set_grad_enabled(False)

from redif.model import Model, ftom
from redif.dataset import Dataset

from effortless_config import Config
from os import path, makedirs
from glob import glob
import soundfile as sf

import numpy as np
from tqdm import tqdm


class args(Config):
    CKPT = None
    DDSP = None
    OUT = "redif_samples/"
    DATA = None
    SAMPLE_LENGTH = 512
    N_SAMPLE = 10
    OFFSET = 0


args.parse_args()

out_dir = path.join(args.OUT, path.basename(path.normpath(args.DATA)))

makedirs(out_dir, exist_ok=True)

n_sample = len(glob(path.join(out_dir, "*.wav")))

# INSTANCIATE MODELS
model = Model.load_from_checkpoint(args.CKPT, strict=False).eval()
model.set_noise_schedule()
ddsp = torch.jit.load(args.DDSP).eval()

# CREATE DATASET
data = Dataset(args.SAMPLE_LENGTH, args.DATA)

for i in tqdm(range(args.N_SAMPLE), desc="sampling"):
    # LOAD DATA
    pitch, loudness, _, _ = data[i + args.OFFSET]
    pitch = pitch.unsqueeze(0)
    loudness = loudness.unsqueeze(0)

    # FORMAT INPUT CONTOURS
    pitch = nn.functional.one_hot(pitch, 127).permute(0, 2, 1)
    loudness = loudness.unsqueeze(1)

    assert torch.all(pitch >= 0) and torch.all(pitch < 128), str(pitch)

    env = torch.cat([pitch, loudness], 1)
    x = torch.randn(env.shape[0], model.data_dim, env.shape[-1])

    # SAMPLE FROM ESTIMATED DISTRIBUTION
    y = model.sample(x, env)
    f0, lo = model.transform.inverse(y)

    # SYNTHESIS
    sound = ddsp(f0.permute(0, 2, 1), lo.permute(0, 2, 1))
    name = f"sample_{i + n_sample:03d}.wav"
    sf.write(path.join(out_dir, name), sound.reshape(-1).numpy(), 16000)

    name = f"env_{i + n_sample:03d}.npy"
    env = np.stack([f0.reshape(-1).numpy(), lo.reshape(-1).numpy()], 0)
    np.save(path.join(out_dir, name), env)

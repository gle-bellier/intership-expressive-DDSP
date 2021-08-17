import librosa as li
from effortless_config import Config
import soundfile as sf
from os import path, makedirs
import tensorboard.backend.event_processing.event_accumulator as ea
from tqdm import tqdm
import numpy as np


class args(Config):
    LOG = None
    OUT = "samples"


args.parse_args()
makedirs(args.OUT, exist_ok=True)

logs = ea.EventAccumulator(args.LOG, size_guidance={ea.AUDIO: 0})
logs.Reload()
audios = logs.Audio("synth")[::-1]


def load_bytes(stream):
    with open("/tmp/audio.wav", "wb") as audio:
        audio.write(stream)
    return li.load("/tmp/audio.wav", None)


def all_valid(x):
    mean = np.mean(x, -1)
    return all(mean != 0)


selected_x = None

for step in tqdm(audios, desc="parsing logs"):
    x, sr = load_bytes(step.encoded_audio_string)
    x = x.reshape(-1, 512 * 160)

    if selected_x is None:
        selected_x = x

    else:
        for i in range(selected_x.shape[0]):
            if np.mean(selected_x[i]) == 0:
                selected_x[i] = x[i]

        if all_valid(selected_x):
            break

for i, x in enumerate(selected_x):
    sf.write(path.join(args.OUT, f"sample_{i:03d}.wav"), x, sr)

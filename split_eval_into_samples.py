import librosa as li
from effortless_config import Config
import soundfile as sf
from os import path, makedirs


class args(Config):
    WAV = None
    OUT = "split_out"


args.parse_args()
makedirs(args.OUT, exist_ok=True)

x, sr = li.load(args.WAV, None)
x = x.reshape(-1, 512 * 160)

for i, sample in enumerate(x):
    sf.write(path.join(args.OUT, f"sample_{i:03d}.wav"), sample, sr)

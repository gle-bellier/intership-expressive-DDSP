from redif.model import Model
from redif.dataset import Dataset
from redif.ema import EMAModelCheckPoint
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from pytorch_lightning import loggers as pl_loggers

import pytorch_lightning as pl
from effortless_config import Config


class args(Config):
    CKPT = None
    DDSP = "ddsp_violin_pretrained.ts"
    DATA = None
    VALIDATION = None


args.parse_args()

# CREATE AND SPLIT DATASET
train = Dataset(256, args.DATA)
val = ConcatDataset(
    [Dataset(512, path) for path in args.VALIDATION.split(",")])

# INSTANCIATE MODEL
if args.CKPT is not None:
    model = Model.load_from_checkpoint(args.CKPT, strict=False)
else:
    model = Model(2, 128, [128, 256, 384])
model.set_noise_schedule()
model.transform.compute_stats(train.e_f0, train.e_lo)
model.ddsp = torch.jit.load(args.DDSP)

# INSTANCIATE TRAINER
tb_logger = pl_loggers.TensorBoardLogger('logs/rediff')
trainer = pl.Trainer(gpus=1,
                     check_val_every_n_epoch=10,
                     callbacks=[EMAModelCheckPoint(
                         model,
                         filename="ema",
                     )],
                     max_epochs=100000,
                     logger=tb_logger,
                     resume_from_checkpoint=args.CKPT)

# FIT
trainer.fit(
    model,
    DataLoader(train, 16, True, drop_last=True),
    DataLoader(val, 16, False, drop_last=False),
)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import pytorch_lightning as pl
import pickle
from random import randint, sample
from utils import *


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, norm=True, act=True):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.norm = nn.LayerNorm(out_size) if norm else None
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act:
            x = nn.functional.leaky_relu(x)
        return x


class ModelContinuousPitch(pl.LightningModule):
    def __init__(self, in_size, hidden_size, out_size, scalers):
        super().__init__()
        self.save_hyperparameters()
        self.scalers = scalers
        self.loudness_nbins = 30
        self.ddsp = torch.jit.load("results/ddsp_debug_pretrained.ts").eval()

        self.pre_lstm = nn.Sequential(
            LinearBlock(in_size, hidden_size),
            LinearBlock(hidden_size, hidden_size),
        )

        self.lstm = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.post_lstm = nn.Sequential(
            LinearBlock(hidden_size, hidden_size),
            LinearBlock(
                hidden_size,
                out_size,
                norm=False,
                act=False,
            ),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=1e-4,
            weight_decay=.01,
        )

    def forward(self, x):
        x = self.pre_lstm(x)
        x = self.lstm(x)[0]
        x = self.post_lstm(x)
        return x

    def split_predictions(self, prediction):
        pred_f0 = prediction[..., :1]
        pred_cents = prediction[..., 1:2]
        pred_loudness = prediction[..., 2:]
        return pred_f0, pred_cents, pred_loudness

    def mse_and_ce(self, pred_f0, pred_cents, pred_loudness, target_f0,
                   target_cents, target_loudness):

        pred_loudness = pred_loudness.permute(0, 2, 1)
        target_loudness = target_loudness.squeeze(-1)

        loss_f0 = nn.functional.mse_loss(pred_f0, target_f0)
        loss_cents = nn.functional.mse_loss(pred_cents, target_cents)
        loss_loudness = nn.functional.cross_entropy(
            pred_loudness,
            target_loudness.long(),
        )

        return loss_f0 * 100, loss_cents * 100, loss_loudness

    def training_step(self, batch, batch_idx):
        model_input, target = batch
        prediction = self.forward(model_input.float())

        pred_f0, pred_cents, pred_loudness = self.split_predictions(prediction)
        target_f0, target_cents, target_loudness = torch.split(target, 1, -1)
        loss_f0, loss_cents, loss_loudness = self.mse_and_ce(
            pred_f0,
            pred_cents,
            pred_loudness,
            target_f0,
            target_cents,
            target_loudness,
        )

        self.log("loss_f0", loss_f0)
        self.log("loss_cents", loss_cents)
        self.log("loss_loudness", loss_loudness)

        return loss_f0 + loss_cents + loss_loudness

    def sample_one_hot(self, x):
        n_bin = x.shape[-1]
        sample = torch.distributions.Categorical(logits=x).sample()
        sample = nn.functional.one_hot(sample, n_bin)
        return sample

    @torch.no_grad()
    def generation_loop(self, x, infer_pitch=False):
        context = None

        for i in range(x.shape[1] - 1):
            x_in = x[:, i:i + 1]

            x_out = self.pre_lstm(x_in)
            x_out, context = self.lstm(x_out, context)
            x_out = self.post_lstm(x_out)
            pred_f0, pred_cents, pred_loudness = self.split_predictions(x_out)

            if infer_pitch:
                f0 = pred_f0
            else:
                f0 = x[:, i + 1:i + 2, :1].float()

            loudness = self.sample_one_hot(pred_loudness)

            cat = torch.cat([f0, pred_cents, loudness], -1)
            ndim = cat.shape[-1]

            x[:, i + 1:i + 2, -ndim:] = cat

        pred = x[..., -ndim:]
        pred_f0, pred_cents, pred_loudness = self.split_predictions(pred)

        pred_f0 = pred_f0[:, 1:]
        pred_loudness = pred_loudness[:, 1:]
        pred_cents = pred_cents[:, :-1]

        out = [pred_f0, pred_cents, torch.argmax(pred_loudness, -1)]

        return list(out)

    def apply_inverse_transform(self, x, idx):
        scaler = self.scalers[idx]
        x = x.cpu()
        out = scaler.inverse_transform(x.reshape(-1, 1))
        out = torch.from_numpy(out).to("cuda")
        out = out.unsqueeze(0)
        return out.float()

    def get_audio(self, model_input, target):

        model_input = model_input.unsqueeze(0).float()
        f0, cents, loudness = self.generation_loop(model_input)
        cents = cents - .5

        f0 = pctof(f0, cents)

        loudness = loudness / (self.loudness_nbins - 1)
        f0 = self.apply_inverse_transform(f0.squeeze(0), 0)
        loudness = self.apply_inverse_transform(loudness.squeeze(0), 1)
        y = self.ddsp(f0, loudness)
        return y

    def validation_step(self, batch, batch_idx):
        model_input, target = batch
        prediction = self.forward(model_input.float())

        pred_f0, pred_cents, pred_loudness = self.split_predictions(prediction)
        target_f0, target_cents, target_loudness = torch.split(target, 1, -1)

        loss_f0, loss_cents, loss_loudness = self.mse_and_ce(
            pred_f0,
            pred_cents,
            pred_loudness,
            target_f0,
            target_cents,
            target_loudness,
        )

        self.log("val_loss_f0", loss_f0)
        self.log("val_loss_cents", loss_cents)
        self.log("val_loss_loudness", loss_loudness)
        self.log("val_total", loss_f0 + loss_cents + loss_loudness)

        ## Every 100 epochs : produce audio

        if self.current_epoch % 20 == 0:

            audio = self.get_audio(model_input[0], target[0])
            # output audio in Tensorboard
            tb = self.logger.experiment
            n = "Epoch={}".format(self.current_epoch)
            tb.add_audio(tag=n, snd_tensor=audio, sample_rate=16000)

            # TODO : CHANGE THIS VISUALISATION BUG
            for i in range(model_input.shape[1]):
                tb.add_scalars("f0", {
                    "model": model_input[0, i, 0],
                    "target": target[0, i, 0],
                }, i)
                tb.add_scalars("cents", {
                    "model": model_input[0, i, 1],
                    "target": target[0, i, 1]
                }, i)


# if __name__ == "__main__":

#     trainer = pl.Trainer(
#         gpus=1,
#         callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_total")],
#         max_epochs=50000,
#     )
#     list_transforms = [
#         (MinMaxScaler, ),  # u_f0
#         (MinMaxScaler, ),  # u_loudness
#         (MinMaxScaler, ),  # e_f0
#         (Identity, ),  # e_cents
#         (MinMaxScaler, ),  # e_loudness
#     ]

#     dataset = ExpressiveDatasetPitchContinuous(n_sample=512,
#                                                list_transforms=list_transforms)
#     val_len = len(dataset) // 20
#     train_len = len(dataset) - val_len

#     train, val = random_split(dataset, [train_len, val_len])

#     model = ModelContinuousPitch(63, 1024, 32, scalers=dataset.scalers)

#     trainer.fit(
#         model,
#         DataLoader(train, 32, True),
#         DataLoader(val, 32),
#     )

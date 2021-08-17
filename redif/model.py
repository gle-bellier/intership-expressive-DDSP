import torch
import torch.nn as nn

import pytorch_lightning as pl

from .blocks import UpBlock, FilmBlock, DownBlock
from .diffusion import DiffusionModel
import math


def ftom(f):
    return 12 * (torch.log(f) - math.log(440)) / math.log(2) + 69


def mtof(m):
    return 440 * 2**((m - 69) / 12)


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean_f0", torch.tensor(0.))
        self.register_buffer("var_f0", torch.tensor(0.))
        self.register_buffer("mean_lo", torch.tensor(0.))
        self.register_buffer("var_lo", torch.tensor(0.))

    def compute_stats(self, f0: torch.Tensor, lo: torch.Tensor):
        f0 = ftom(f0)
        self.mean_f0 = f0.mean()
        self.var_f0 = f0.var()

        self.mean_lo = lo.mean()
        self.var_lo = lo.var()

    @torch.no_grad()
    def forward(self, f0, lo):
        f0 = ftom(f0)
        f0 = (f0 - self.mean_f0) / self.var_f0

        lo = (lo - self.mean_lo) / self.var_lo
        return torch.stack([f0, lo], 1)

    @torch.no_grad()
    def inverse(self, x):
        f0, lo = torch.split(x, 1, 1)

        lo = lo * self.var_lo + self.mean_lo
        f0 = f0 * self.var_f0 + self.mean_f0
        f0 = mtof(f0)
        return f0, lo


class Model(pl.LightningModule, DiffusionModel):
    def __init__(self, data_dim, cdt_dim, dims):
        super().__init__()
        self.save_hyperparameters()

        self.transform = Transform()
        self.ddsp = None
        self.val_step = 0
        self.data_dim = data_dim
        self.cdt_dim = cdt_dim

        # DOWN CHAIN
        ndc = [nn.Conv1d(data_dim, dims[0], 3, padding=1)]
        edc = [nn.Conv1d(cdt_dim, dims[0], 3, padding=1)]

        for i in range(len(dims) - 1):
            ndc.append(DownBlock(dims[i], dims[i + 1]))
            edc.append(DownBlock(dims[i], dims[i + 1]))

        self.ndc = nn.ModuleList(ndc)
        self.edc = nn.ModuleList(edc)

        # FILM CHAIN
        nfc = []
        efc = []

        for i in range(len(dims)):
            nfc.append(FilmBlock(dims[i], dims[i]))
            efc.append(FilmBlock(dims[i], dims[i]))

        self.nfc = nn.ModuleList(nfc)
        self.efc = nn.ModuleList(efc)

        # UP CHAIN
        dims = [dims[-1] * 2] + dims[::-1]

        uc = []

        for i in range(len(dims) - 1):
            uc.append(UpBlock(
                dims[i],
                dims[i + 1],
                dims[i + 1],
                bool(i),
            ))

        self.uc = nn.ModuleList(uc)

        self.post_process = nn.Conv1d(dims[-1], data_dim, 3, padding=1)

    def neural_pass(self, y, env, noise_level):
        ndc = []
        edc = []

        for noise_layer, envelop_layer in zip(self.ndc, self.edc):
            y = noise_layer(y)
            env = envelop_layer(env)

            ndc.append(y)
            edc.append(env)

        x = torch.cat([ndc[-1], edc[-1]], 1)

        nfc = []
        efc = []

        for noise_layer, envelop_layer in zip(self.nfc, self.efc):
            nfc.append(noise_layer(ndc.pop(0), noise_level))
            efc.append(envelop_layer(edc.pop(0), noise_level))

        for layer in self.uc:
            x = layer(x, efc.pop(-1), nfc.pop(-1))

        return self.post_process(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        u_f0, u_lo, e_f0, e_lo = batch

        # APPLY TRANSFORM ON F0 AND LO
        x = self.transform(e_f0, e_lo)

        # QUANTIZED PITCH -> ONE HOT AND CAT PITCH AND LOUDNESS
        u_f0 = nn.functional.one_hot(u_f0, self.cdt_dim - 1).permute(0, 2, 1)
        env = torch.cat([u_f0, u_lo.unsqueeze(1)], 1)

        loss = self.compute_loss(x, env)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        u_f0, u_lo, e_f0, e_lo = batch

        # APPLY TRANSFORM ON F0 AND LO
        x = self.transform(e_f0, e_lo)

        # QUANTIZED PITCH -> ONE HOT AND CAT PITCH AND LOUDNESS
        u_f0 = nn.functional.one_hot(u_f0, 127).permute(0, 2, 1)
        env = torch.cat([u_f0, u_lo.unsqueeze(1)], 1)

        self.log("validation", self.compute_loss(x, env))

        return env

    def validation_epoch_end(self, out):
        env = torch.cat(out, 0)
        x = torch.randn(env.shape[0], self.data_dim, env.shape[-1]).to(env)

        y = self.sample(x, env)  # GENERATE CONTOURS
        f0, lo = self.transform.inverse(y)  # INVERSE TRANSFORM

        if self.ddsp is not None:
            y = self.ddsp(
                f0.permute(0, 2, 1),
                lo.permute(0, 2, 1),
            )  # SYNTH AUDIO
            self.logger.experiment.add_audio(
                "synth",
                y.reshape(-1),
                self.val_step,
                16000,
            )

        self.val_step += 1

    @torch.no_grad()
    def sample(self, x, env):
        x = torch.randn_like(x)
        for i in range(self.n_step)[::-1]:
            x = self.inverse_dynamics(x, env, i, clip=False)
        return x

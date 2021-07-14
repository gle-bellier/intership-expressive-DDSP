import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from torch import nn
from utils import *
from downsampling import DBlock
from upsampling import UBlock
from diffusion import DiffusionModel
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from diffusion_dataset import DiffusionDataset
import matplotlib.pyplot as plt
import math


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lr = nn.LeakyReLU()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.mp = nn.MaxPool1d(kernel_size=2)
        pass

    def forward(self, x):
        x = self.conv1(x)
        ctx = torch.clone(x)
        x = self.conv2(x)
        out = self.mp(x)
        return out, ctx


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.gru = nn.GRU(input_size=out_channels,
                          hidden_size=out_channels,
                          batch_first=True)

    def forward(self, x):

        x = self.conv1(x)

        out = self.conv2(x)
        return out


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(2 * out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               stride=1,
                               kernel_size=3,
                               padding=1))
        self.conv_ctx = nn.ConvTranspose1d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           stride=1,
                                           kernel_size=3,
                                           padding=1)

        self.lr = nn.LeakyReLU()

    def add_ctx(self, x, ctx):
        # # crop context (y)
        # d_shape = (ctx.shape[-1] - x.shape[-1]) // 2
        # crop = ctx[:, :, d_shape:d_shape + x.shape[2]]
        # #concatenate
        out = torch.cat([x, ctx], 1)
        return out

    def forward(self, x, ctx):
        x = self.up_conv(x)
        ctx = self.conv_ctx(ctx)
        x = self.add_ctx(x, ctx)
        x = self.conv1(x)
        out = self.conv2(x)
        return out


class UNet(pl.LightningModule):
    def __init__(self, channels, scalers, ddsp):
        super().__init__()
        #self.save_hyperparameters()

        down_channels = channels
        up_channels = channels[::-1]

        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]

        self.up_channels_in = up_channels[:-1]
        self.up_channels_out = up_channels[1:]

        self.scalers = scalers
        self.ddsp = ddsp
        self.val_idx = 0

        self.down_blocks = nn.ModuleList([
            DBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.down_channels_in,
                                                 self.down_channels_out)
        ])

        self.bottleneck = Bottleneck(in_channels=self.down_channels_out[-1],
                                     out_channels=self.up_channels_in[0])

        self.up_blocks = nn.ModuleList([
            UBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.up_channels_in,
                                                 self.up_channels_out)
        ])

    def down_sampling(self, x):
        l_ctx = []
        for i in range(len(self.down_blocks)):
            x, ctx = self.down_blocks[i](x)
            l_ctx = [ctx] + l_ctx

        return x, l_ctx

    def up_sampling(self, x, l_ctx):

        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x, l_ctx[i])
        return x

    def forward(self, x):
        # permute from B, T, C -> B, C, T
        x = x.permute(0, 2, 1)
        out, l_ctx = self.down_sampling(x)
        out = self.bottleneck(out)
        out = self.up_sampling(out, l_ctx)
        # permute from B, C, T -> B, T, C
        out = out.permute(0, 2, 1)

        return out


class UNet_Diffusion(pl.LightningModule, DiffusionModel):
    def __init__(self, channels, scalers, ddsp):
        super().__init__()
        #self.save_hyperparameters()
        self.channels = channels

        self.scalers = scalers
        self.ddsp = ddsp
        self.val_idx = 0

        self.unet = UNet(channels, scalers, ddsp)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    def neural_pass(self, y, cdt, noise_level):
        out = self.unet(y)
        return out

    def training_step(self, batch, batch_idx):
        model_input, cdt = batch
        loss = self.compute_loss(model_input, cdt)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        # loss = self.compute_loss(batch, batch_idx) Why ??

        model_input, cdt = batch
        loss = self.compute_loss(model_input, cdt)
        self.log("val_loss", loss)

        # returns cdt for validation end epoch
        return cdt

    def post_process(self, out):

        # change range [-1, 1] -> [0, 1]
        out = out / 2 + .5

        f0, l0 = torch.split(out, 1, -1)
        f0 = f0.reshape(-1, 1).cpu().numpy()
        l0 = l0.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        l0 = self.scalers[1].inverse_transform(l0).reshape(-1)

        return f0, l0

    def validation_epoch_end(self, cdt):

        # test for last cdt
        cdt = cdt[-1]

        self.val_idx += 1

        if self.val_idx % 50:
            return

        device = next(iter(self.parameters())).device

        out = self.partial_denoising(cdt, cdt, 30)

        f0, lo = out[0].split(1, -1)

        plt.plot(f0.cpu())
        self.logger.experiment.add_figure("pitch RAW", plt.gcf(), self.val_idx)
        plt.plot(lo.cpu())
        self.logger.experiment.add_figure("loudness RAW", plt.gcf(),
                                          self.val_idx)

        # select first elt :

        f0, lo = self.post_process(out[0])

        plt.plot(f0)
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(lo)
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)

        if self.ddsp is not None:
            f0 = torch.from_numpy(f0).float().reshape(1, -1, 1).to("cuda")
            lo = torch.from_numpy(lo).float().reshape(1, -1, 1).to("cuda")
            signal = self.ddsp(f0, lo)
            signal = signal.reshape(-1).cpu().numpy()

            self.logger.experiment.add_audio(
                "generation",
                signal,
                self.val_idx,
                16000,
            )

    @torch.no_grad()
    def sample(self, x, cdt):
        x = torch.randn_like(x)
        for i in range(self.n_step)[::-1]:
            x = self.inverse_dynamics(x, cdt, i)
        return x

    @torch.no_grad()
    def partial_denoising(self, x, cdt, n_step):
        noise_level = self.sqrt_alph_cum_prev[n_step]
        eps = torch.randn_like(x)
        x = noise_level * x
        x = x + math.sqrt(1 - noise_level**2) * eps

        for i in range(n_step)[::-1]:
            x = self.inverse_dynamics(x, cdt, i)
        return x


if __name__ == "__main__":
    tb_logger = pl_loggers.TensorBoardLogger('logs/diffusion/')

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        max_epochs=100000,
        logger=tb_logger)
    list_transforms = [
        (MinMaxScaler, ),
        (QuantileTransformer, 30),
    ]

    dataset = DiffusionDataset(list_transforms=list_transforms)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    channels = [2, 8, 128, 512]
    

    ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

    model = UNet_Diffusion(scalers=dataset.scalers,
                           channels = channels,
                           ddsp=ddsp)

    model.set_noise_schedule()

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )

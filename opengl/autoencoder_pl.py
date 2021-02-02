import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torchvision import datasets, transforms
from PIL import Image
import torchvision
import numpy as np
from BBCE import BootstrappedCE
from conv_ae import HardMish

class ConvAutoencoder5Logits(pl.LightningModule):
    def __init__(self, hparams, size=3, bn=False):
        super(ConvAutoencoder5Logits, self).__init__()
        ## encoder layers ##
        self.hparams = hparams
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.learning_rate = hparams.learning_rate
        self.criterion = BootstrappedCE(3, 0, 1.0 / 9.0)
        self.activation = HardMish()
        self.bn = bn
        self.conv11 = nn.Conv2d(size, 128, 5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 256, 5, stride=2)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv31 = nn.Conv2d(256, 512, 5, stride=2)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv41 = nn.Conv2d(512, 512, 5, stride=2)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.dense_enc = nn.Linear(512*5*5, 128)
        self.dense_enc_bn = nn.BatchNorm1d(128)

        h = w = 128
        self._strides = [2, 2, 2, 2]
        self._num_filters = [512, 512, 256, 128]

        layer_dimensions = [[int(h/np.prod(self._strides[i:])), int(w/np.prod(self._strides[i:]))]
                            for i in range(len(self._strides))]
        self._layer_dimensions = layer_dimensions
        self.dense = nn.Linear(128, layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0])

        self.t_convs = []
        # w = (w-k+2*p)/S+1
        # S*(w-1)-w+k =2*p
        # 1*7-8+3
        # decoder layers
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.dense_bn = nn.BatchNorm2d(512)
        self.t_conv11 = nn.Conv2d(512, 512, 5, padding=2)
        self.t_conv1_bn = nn.BatchNorm2d(512)
        self.t_conv21 = nn.Conv2d(512, 256, 5, padding=2)
        self.t_conv2_bn = nn.BatchNorm2d(256)
        self.t_conv31 = nn.Conv2d(256, 128, 5, padding=2)
        self.t_conv3_bn = nn.BatchNorm2d(128)
        self.t_conv41 = nn.Conv2d(128, 128, 5, padding=2)
        self.t_conv4_bn = nn.BatchNorm2d(1)
        self.t_conv51 = nn.Conv2d(128, 128, 3, padding=1)
        self.t_conv52 = nn.Conv2d(128, size, 3, padding=1)

    def forward(self, x):
        # encode
        # add hidden layers with relu activation function
        # and maxpooling after

        x = self.encode(x)

        x = self.decode(x)

        return x

    def encode(self, x):
        x = self.activation(self.conv11(x))
        if self.bn:
            x = self.conv1_bn(x)
        # add second hidden layer
        x = self.activation(self.conv21(x))
        if self.bn:
            x = self.conv2_bn(x)
        # add second hidden layer
        x = self.activation(self.conv31(x))
        if self.bn:
            x = self.conv3_bn(x)
        # add second hidden layer
        x = self.activation(self.conv41(x))
        if self.bn:
            x = self.conv4_bn(x)

        x = x.reshape(-1, 12800)
        x = self.activation(self.dense_enc(x))
        if self.bn:
            x = self.dense_enc_bn(x)
        return x

    def decode(self, x):
        x = self.activation(self.dense(x))
        #
        # x = F.relu(x)
        x = x.view(-1, self._num_filters[0], self._layer_dimensions[0][0], self._layer_dimensions[0][1])
        # ## decode ##
        # add transpose conv layers, with relu activation function
        if self.bn:
            x = self.dense_bn(x)

        x = self.activation(self.t_conv11(x))
        if self.bn:
            x = self.t_conv1_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[1])

        x = self.activation(self.t_conv21(x))
        if self.bn:
            x = self.t_conv2_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[2])

        x = self.activation(self.t_conv31(x))
        if self.bn:
            x = self.t_conv3_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[3])

        x = self.activation(self.t_conv41(x))
        if self.bn:
            x = self.t_conv4_bn(x)

        x = nn.functional.interpolate(x, [128, 128])

        x = self.activation(self.t_conv51(x))
        x = self.t_conv52(x)
        return x

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), self.learning_rate)
        ds_len = 20224  # check in train loader?
        total_bs = self.hparams.batch_size*self.hparams.gpus
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                        self.optimizer, max_lr=self.learning_rate,
                                        anneal_strategy='linear', div_factor=self.hparams.one_cycle_div_factor,
                                        steps_per_epoch=int((ds_len/total_bs)),
                                        epochs=self.hparams.max_epochs)
        sched = {
            'scheduler': self.scheduler,
            'interval': 'step',
        }
        return [self.optimizer], [sched]

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss, top_k = self.criterion(x_hat, x)
        return {'loss': loss, 'lr': self.learning_rate}

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        # mu, logvar = self.encode(x.view(-1, 784))
        # z = self.reparameterize(mu, logvar)
        x_hat = self(x)
        val_loss, top_k = self.criterion(x_hat, x)

        return {'val_loss': val_loss, 'x_hat': x_hat}

    def validation_epoch_end(
            self,
            outputs):

        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['x_hat']
        x_hat = torch.sigmoid(x_hat)
        grid = torchvision.utils.make_grid(x_hat, normalize=True)

        self.logger.experiment.add_image('images', grid, self.current_epoch)

        self.log('avg_val_loss', val_loss)

    def train_dataloader(self):
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(root='../datasets/autoencoder/train',
                                                   transform=data_transform)
        # dataload same for in and out
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True,
                                             num_workers=self.hparams.num_workers)
        return train_loader

    def val_dataloader(self):
        # dataload same for in and out
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(root='../datasets/autoencoder/val', transform=data_transform)
        # dataload same for in and out
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size,
                                                 num_workers=self.hparams.num_workers)
        return val_loader


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=6e-4, type=float)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--one_cycle_div_factor', default=25, type=int)

    args = parser.parse_args()

    vae = ConvAutoencoder5Logits(hparams=args)
    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer.fit(vae)
    # lr_finder = trainer.tuner.lr_find(vae, max_lr=1e-3, min_lr=1e-6)
    #
    # # Results can be found in
    # lr_finder.results
    #
    # # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(new_lr)

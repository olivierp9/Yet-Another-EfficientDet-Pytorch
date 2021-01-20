import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, tanh


def hard_mish(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, bn=False):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.bn = bn
        self.conv1 = nn.Conv2d(1, 128, 5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 5, stride=2)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 5, stride=2)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 5, stride=2)
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

        # decoder layers
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.dense_bn = nn.BatchNorm2d(512)
        self.t_conv1 = nn.Conv2d(512, 512, 5, padding=2)
        self.t_conv1_bn = nn.BatchNorm2d(512)
        self.t_conv2 = nn.Conv2d(512, 256, 5, padding=2)
        self.t_conv2_bn = nn.BatchNorm2d(256)
        self.t_conv3 = nn.Conv2d(256, 128, 5, padding=2)
        self.t_conv3_bn = nn.BatchNorm2d(128)
        self.t_conv4 = nn.Conv2d(128, 128, 5, padding=2)
        self.t_conv4_bn = nn.BatchNorm2d(1)
        self.t_conv5 = nn.Conv2d(128, 1, 5, padding=2)

    def forward(self, x):
        # encode
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        if self.bn:
            x = self.conv1_bn(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        if self.bn:
            x = self.conv2_bn(x)
        # add second hidden layer
        x = F.relu(self.conv3(x))
        if self.bn:
            x = self.conv3_bn(x)
        # add second hidden layer
        x = F.relu(self.conv4(x))
        if self.bn:
            x = self.conv4_bn(x)

        x = x.view(-1, 12800)
        x = F.relu(self.dense_enc(x))
        if self.bn:
            x = self.dense_enc_bn(x)

        x = F.relu(self.dense(x))
        #
        # x = F.relu(x)
        x = x.view(-1, self._num_filters[0], self._layer_dimensions[0][0], self._layer_dimensions[0][1])
        # ## decode ##
        # add transpose conv layers, with relu activation function
        if self.bn:
            x = self.dense_bn(x)

        x = F.relu(self.t_conv1(x))
        if self.bn:
            x =self.t_conv1_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[1])

        x = F.relu(self.t_conv2(x))
        if self.bn:
            x =self.t_conv2_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[2])

        x = F.relu(self.t_conv3(x))
        if self.bn:
            x =self.t_conv3_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[3])

        x = F.relu(self.t_conv4(x))
        if self.bn:
            x =self.t_conv4_bn(x)

        x = nn.functional.interpolate(x, [128, 128])

        x = tanh(self.t_conv5(x))

        return x


# define the NN architecture
class ConvAutoencoder3(nn.Module):
    def __init__(self, size=3, bn=False):
        super(ConvAutoencoder3, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.activation = HardMish()
        self.bn = bn
        self.conv11 = nn.Conv2d(size, 128, 3, stride=1)
        self.conv12 = nn.Conv2d(128, 128, 3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 256, 3, stride=1)
        self.conv22 = nn.Conv2d(256, 256, 3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv31 = nn.Conv2d(256, 512, 3, stride=1)
        self.conv32 = nn.Conv2d(512, 512, 3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv41 = nn.Conv2d(512, 512, 3, stride=1)
        self.conv42 = nn.Conv2d(512, 512, 3, stride=2)
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
        self.t_conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.t_conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(512)
        self.t_conv21 = nn.Conv2d(512, 512, 3, padding=1)
        self.t_conv22 = nn.Conv2d(512, 256, 3, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(256)
        self.t_conv31 = nn.Conv2d(256, 256, 3, padding=1)
        self.t_conv32 = nn.Conv2d(256, 128, 3, padding=1)
        self.t_conv3_bn = nn.BatchNorm2d(128)
        self.t_conv41 = nn.Conv2d(128, 128, 3, padding=1)
        self.t_conv42 = nn.Conv2d(128, 128, 3, padding=1)
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
        x = self.activation(self.conv12(x))
        if self.bn:
            x = self.conv1_bn(x)
        # add second hidden layer
        x = self.activation(self.conv21(x))
        x = self.activation(self.conv22(x))
        if self.bn:
            x = self.conv2_bn(x)
        # add second hidden layer
        x = self.activation(self.conv31(x))
        x = self.activation(self.conv32(x))
        if self.bn:
            x = self.conv3_bn(x)
        # add second hidden layer
        x = self.activation(self.conv41(x))
        x = self.activation(self.conv42(x))
        if self.bn:
            x = self.conv4_bn(x)

        x = x.view(-1, 12800)
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
        x = self.activation(self.t_conv12(x))
        if self.bn:
            x = self.t_conv1_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[1])

        x = self.activation(self.t_conv21(x))
        x = self.activation(self.t_conv22(x))
        if self.bn:
            x = self.t_conv2_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[2])

        x = self.activation(self.t_conv31(x))
        x = self.activation(self.t_conv32(x))
        if self.bn:
            x = self.t_conv3_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[3])

        x = self.activation(self.t_conv41(x))
        x = self.activation(self.t_conv42(x))
        if self.bn:
            x = self.t_conv4_bn(x)

        x = nn.functional.interpolate(x, [128, 128])

        x = self.activation(self.t_conv51(x))
        x = sigmoid(self.t_conv52(x))
        return x


# define the NN architecture
class ConvAutoencoderT(nn.Module):
    def __init__(self, bn=False):
        super(ConvAutoencoderT, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.bn = bn
        self.conv1 = nn.Conv2d(1, 128, 5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 5, stride=2)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 5, stride=2)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 5, stride=2)
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
        self.dense_up = nn.Conv2d(512, 512, 5, stride=2)

        # decoder layers
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.dense_bn = nn.BatchNorm2d(512)
        self.t_conv1 = nn.Conv2d(512, 512, 5, padding=2)
        self.t_up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.t_conv1_bn = nn.BatchNorm2d(512)
        self.t_conv2 = nn.Conv2d(512, 256, 5, padding=2)
        self.t_up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.t_conv2_bn = nn.BatchNorm2d(256)
        self.t_conv3 = nn.Conv2d(256, 128, 5, padding=2)
        self.t_up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.t_conv3_bn = nn.BatchNorm2d(128)
        self.t_conv4 = nn.Conv2d(128, 1, 5, padding=2)
        self.t_up4 = nn.ConvTranspose2d(1, 1, 2, stride=2)
        self.t_conv4_bn = nn.BatchNorm2d(1)
        self.t_conv5 = nn.Conv2d(1, 1, 5, padding=2)

    def forward(self, x):
        # encode
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        if self.bn:
            x = self.conv1_bn(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        if self.bn:
            x = self.conv2_bn(x)
        # add second hidden layer
        x = F.relu(self.conv3(x))
        if self.bn:
            x = self.conv3_bn(x)
        # add second hidden layer
        x = F.relu(self.conv4(x))
        if self.bn:
            x = self.conv4_bn(x)

        x = x.view(-1, 12800)
        x = F.relu(self.dense_enc(x))
        if self.bn:
            x = self.dense_enc_bn(x)

        x = F.relu(self.dense(x))
        #
        # x = F.relu(x)
        x = x.view(-1, self._num_filters[0], self._layer_dimensions[0][0], self._layer_dimensions[0][1])
        # ## decode ##
        # add transpose conv layers, with relu activation function
        if self.bn:
            x = self.dense_bn(x)

        x = F.relu(self.t_conv1(x))
        if self.bn:
            x =self.t_conv1_bn(x)
        x = self.t_up1(x)

        x = F.relu(self.t_conv2(x))
        if self.bn:
            x =self.t_conv2_bn(x)
        x = self.t_up2(x)

        x = F.relu(self.t_conv3(x))
        if self.bn:
            x =self.t_conv3_bn(x)
        x = self.t_up3(x)

        x = F.relu(self.t_conv4(x))
        if self.bn:
            x =self.t_conv4_bn(x)
        x = self.t_up4(x)

        x = tanh(self.t_conv5(x))

        return x


# define the NN architecture
class ConvAutoencoder32(nn.Module):
    def __init__(self, bn=False):
        super(ConvAutoencoder32, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.bn = bn
        self.conv11 = nn.Conv2d(1, 64, 3, stride=1)
        self.conv12 = nn.Conv2d(64, 64, 3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(64, 128, 3, stride=1)
        self.conv22 = nn.Conv2d(128, 128, 3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv31 = nn.Conv2d(128, 256, 3, stride=1)
        self.conv32 = nn.Conv2d(256, 256, 3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv41 = nn.Conv2d(256, 256, 3, stride=1)
        self.conv42 = nn.Conv2d(256, 256, 3, stride=2)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.dense_enc = nn.Linear(256*5*5, 128)
        self.dense_enc_bn = nn.BatchNorm1d(128)

        h = w = 128
        self._strides = [2, 2, 2, 2]
        self._num_filters = [256, 256, 128, 64]

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
        self.dense_bn = nn.BatchNorm2d(256)
        self.t_conv11 = nn.Conv2d(256, 256, 3, padding=1)
        self.t_conv12 = nn.Conv2d(256, 256, 3, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(256)
        self.t_conv21 = nn.Conv2d(256, 256, 3, padding=1)
        self.t_conv22 = nn.Conv2d(256, 128, 3, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(128)
        self.t_conv31 = nn.Conv2d(128, 128, 3, padding=1)
        self.t_conv32 = nn.Conv2d(128, 64, 3, padding=1)
        self.t_conv3_bn = nn.BatchNorm2d(64)
        self.t_conv41 = nn.Conv2d(64, 64, 3, padding=1)
        self.t_conv42 = nn.Conv2d(64, 64, 3, padding=1)
        self.t_conv4_bn = nn.BatchNorm2d(1)
        self.t_conv51 = nn.Conv2d(64, 64, 3, padding=1)
        self.t_conv52 = nn.Conv2d(64, 1, 3, padding=1)

        self.activation = HardMish()


    def forward(self, x):
        # encode
        # add hidden layers with relu activation function
        # and maxpooling after
        activation = HardMish()

        x = self.activation(self.conv11(x))
        x = self.activation(self.conv12(x))
        if self.bn:
            x = self.conv1_bn(x)
        # add second hidden layer
        x = self.activation(self.conv21(x))
        x = self.activation(self.conv22(x))
        if self.bn:
            x = self.conv2_bn(x)
        # add second hidden layer
        x = self.activation(self.conv31(x))
        x = self.activation(self.conv32(x))
        if self.bn:
            x = self.conv3_bn(x)
        # add second hidden layer
        x = self.activation(self.conv41(x))
        x = self.activation(self.conv42(x))
        if self.bn:
            x = self.conv4_bn(x)

        x = x.view(-1, 256*5*5)
        x = self.activation(self.dense_enc(x))
        if self.bn:
            x = self.dense_enc_bn(x)

        x = self.activation(self.dense(x))
        #
        # x = F.relu(x)
        x = x.view(-1, self._num_filters[0], self._layer_dimensions[0][0], self._layer_dimensions[0][1])
        # ## decode ##
        # add transpose conv layers, with relu activation function
        if self.bn:
            x = self.dense_bn(x)

        x = self.activation(self.t_conv11(x))
        x = self.activation(self.t_conv12(x))
        if self.bn:
            x =self.t_conv1_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[1])

        x = self.activation(self.t_conv21(x))
        x = self.activation(self.t_conv22(x))
        if self.bn:
            x =self.t_conv2_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[2])

        x = self.activation(self.t_conv31(x))
        x = self.activation(self.t_conv32(x))
        if self.bn:
            x =self.t_conv3_bn(x)

        x = nn.functional.interpolate(x, self._layer_dimensions[3])

        x = self.activation(self.t_conv41(x))
        x = self.activation(self.t_conv42(x))
        if self.bn:
            x =self.t_conv4_bn(x)

        x = nn.functional.interpolate(x, [128, 128])

        x = self.activation(self.t_conv51(x))
        x = sigmoid(self.t_conv52(x))

        return x
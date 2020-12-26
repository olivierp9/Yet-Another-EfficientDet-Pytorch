import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from BBCE import BoostStrappedCrossEntropy, BootstrappedCE
from conv_ae import ConvAutoencoder, ConvAutoencoderT, ConvAutoencoder32, ConvAutoencoder3


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = ConvAutoencoder3()
    model = ConvAutoencoder32()
    # model.load_state_dict(torch.load("models/model5000.pth"))
    print(model)
    model.to(device)
    summary(model, (1, 128, 128))

    # Create training and test dataloaders

    num_workers = 0
    # how many samples per batch to load
    batch_size = 48

    target_data = np.load("data/700mm_norm_1.npy")

    for i in range(target_data.shape[0]):
        if np.max(target_data[i,:]) == np.min(target_data[i,:]):
            print(i)
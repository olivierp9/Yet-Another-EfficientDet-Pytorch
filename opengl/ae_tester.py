import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import time
from conv_ae import ConvAutoencoder3


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     #C#reate training and test dataloaders

    num_workers = 0
    # how many samples per batch to load
    batch_size = 1

    train_data = np.load("data/700mm_norm_1_aug.npy")
    train_out = np.load("data/700mm_norm_1.npy")

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)

    model = ConvAutoencoder3()
    model.load_state_dict(torch.load("models_conv3_hmish/model140.pth"))
    model.to(device)
    model.eval()
    i = 0
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images = data
        # clear the gradients of all optimized variables
        images = images.permute(0, 3, 1, 2).float().to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)

        plt.imshow(images.cpu().detach().numpy()[0, 0, :, :], cmap='gray')
        plt.show()

        plt.imshow(outputs.cpu().detach().numpy()[0, 0, :, :], cmap='gray')
        plt.show()

        tmp = train_out[i, :, :, :]

        # B = np.einsum('kij->ijk', tmp)

        # plt.imshow(tmp, cmap='gray')
        # plt.show()
        #
        # i+=1
        print("sleep")
        time.sleep(5)



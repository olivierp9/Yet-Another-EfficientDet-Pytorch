import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from BBCE import BoostStrappedCrossEntropy, BootstrappedCE
from conv_ae import ConvAutoencoder, ConvAutoencoderT, ConvAutoencoder32, ConvAutoencoder3
from imgaug.augmenters import Sequential, Sometimes, CropAndPad, Cutout, Affine, GaussianBlur




# custom dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels=None):
        self.X = images
        self.y = labels

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        if self.y is not None:
            return (self.X[i,:,:,:], self.y[i,:,:,:])
        else:
            return data


if __name__ == "__main__":

    seq = Sequential([
        Sometimes(0.5, Affine(scale=(0.5, 3.0))),
        Sometimes(0.5, CropAndPad(percent=(-0.2, 0.2))),
        Sometimes(0.5, Cutout(nb_iterations=(1, 5), cval=(0.0, 1.0), squared=False, size=0.2, fill_per_channel=True)),
        Sometimes(0.5, Cutout(nb_iterations=(1, 5), cval=(0.0, 1.0), squared=False, size=0.5, fill_per_channel=True,
                              fill_mode="background")),
        Sometimes(0.5, GaussianBlur(sigma=(0.0, 1.2))),
    ], random_order=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = ConvAutoencoder3()
    model = ConvAutoencoder3()
    model.load_state_dict(torch.load("models_normal_map/model20.pth"))
    print(model)
    model.to(device)
    summary(model, (3, 128, 128))

    # Create training and test dataloaders

    num_workers = 0
    # how many samples per batch to load
    batch_size = 16

    # train_data = np.load("data/700mm_norm_1_aug.npy")
    # target_data = np.load("data/700mm_norm_1.npy")
    # print(np.max(train_data), np.min(train_data))
    # print(np.max(target_data), np.min(target_data))
    import gzip
    # f = gzip.GzipFile('data/augz.npy.gz', "r")
    # train_data = np.load(f)
    f = gzip.GzipFile('my_array.npy.gz', "r")
    target_data = np.load(f)
    # img = img[::50]
    # for i in range(img.shape[0]-1,0,-1):
    #     im = img[i, :, :, :]
    #     plt.imshow(im)
    #     plt.show()

    # train_loader = torch.utils.data.DataLoader(CustomDataset(train_data, target_data), batch_size=batch_size,
    #                                            num_workers=num_workers, shuffle=True, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True, pin_memory=True)
    # model.load_state_dict(torch.load("models_test/model660.pth"))

    model.to(device)
    criterion = BootstrappedCE(1, 0, 1.0/9.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    # number of epochs to train the model
    n_epochs = 150
    print("Starting")
    for epoch in range(21, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images = seq(images=data.numpy())
            # images, targets = data
            # clear the gradients of all optimized variables
            images = torch.Tensor(images).permute(0, 3, 1, 2).float().to(device)
            targets = data.permute(0, 3, 1, 2).float().to(device)

            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # outputs = outputs.short()
            # images = images.long()
            # calculate the loss
            loss, _ = criterion(outputs, targets, epoch)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))

        out = outputs[0].cpu().detach().numpy()[:, :, :]
        plt.imshow(np.moveaxis(out, 0, -1))
        plt.show()

        im = targets[0].cpu().detach().numpy()[:, :, :]
        plt.imshow(np.moveaxis(im, 0, -1))
        plt.show()

        im = images[0].cpu().detach().numpy()[:, :, :]
        plt.imshow(np.moveaxis(im, 0, -1))
        plt.show()

        if epoch % 10 == 0:

            torch.save(model.state_dict(), f"models_normal_map/model{epoch}.pth")

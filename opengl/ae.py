import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from BBCE import BoostStrappedCrossEntropy, BootstrappedCE
from conv_ae import ConvAutoencoder, ConvAutoencoderT, ConvAutoencoder32, ConvAutoencoder3


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = ConvAutoencoder3()
    model = ConvAutoencoder3()
    model.load_state_dict(torch.load("models_conv3_hmish/model60.pth"))
    print(model)
    model.to(device)
    summary(model, (1, 128, 128))

    # Create training and test dataloaders

    num_workers = 0
    # how many samples per batch to load
    batch_size = 24

    train_data = np.load("data/700mm_norm_1_aug.npy")
    target_data = np.load("data/700mm_norm_1.npy")
    print(np.max(train_data), np.min(train_data))
    print(np.max(target_data), np.min(target_data))

    train_loader = torch.utils.data.DataLoader(CustomDataset(train_data,target_data), batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True, pin_memory=True)
    # model.load_state_dict(torch.load("models_test/model660.pth"))

    model.to(device)
    criterion = BootstrappedCE(1,0,0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # number of epochs to train the model
    n_epochs = 30000
    print("Starting")
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, targets = data
            # clear the gradients of all optimized variables
            images = images.permute(0, 3, 1, 2).float().to(device)
            targets = targets.permute(0, 3, 1, 2).float().to(device)

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

        # plt.imshow(outputs[0].cpu().detach().numpy()[0, :, :], cmap='gray')
        # plt.show()
        #
        # plt.imshow(targets[0].cpu().detach().numpy()[0, :, :], cmap='gray')
        # plt.show()

        if epoch % 20 == 0:

            torch.save(model.state_dict(), f"models_conv3_hmish/model{epoch+60}.pth")

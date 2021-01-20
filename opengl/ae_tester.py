import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import time
from conv_ae import ConvAutoencoder3
import cv2
import gzip

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     #C#reate training and test dataloaders

    num_workers = 0
    # how many samples per batch to load
    batch_size = 1

    ###

    # train_data = np.load("data/700mm_norm_1_aug.npy")
    # train_out = np.load("data/700mm_norm_1.npy")
    # train_out = np.load("data/700mm_norm_1_aug.npy")
    # mean = np.mean(train_out[train_out>0])
    # std = np.std(train_out[train_out>0])
    ###
    train_data = np.load("../tests_bunny/test3.npy")
    # train_data_0 = train_data>0
    # maxx = np.max(train_data[train_data_0])
    # minn = np.min(train_data[train_data_0])
    # 90340
    # 4.0
    # 21166.0
    #
    # train_data[train_data_0] = (train_data[train_data_0]-minn)/(maxx-minn)
    #
    # mean = np.mean(train_data[train_data_0])
    # std = np.std(train_data[train_data_0])
    # train_data[train_data_0]*=0.5
    # train_data[train_data_0]+=0.5
    # train_data[train_data<0.01] = 0
    # train_data_0 = train_data>0
    # train_data[train_data_0] = 1-train_data[train_data_0]
    top_s = (128-train_data.shape[0])//2
    bottom_s = int(np.ceil((128-train_data.shape[0])/2))
    right_s = (128-train_data.shape[1])//2
    left_s = int(np.ceil((128-train_data.shape[1])/2))
    res = cv2.copyMakeBorder(
        train_data,
        top=top_s,
        bottom=bottom_s,
        left=left_s,
        right=right_s,
        borderType=cv2.BORDER_CONSTANT,
        value=[0]
    )
    # res = cv2.resize(train_data, dsize=(128, 128))
    res = np.expand_dims(res, 0)
    train_out = np.swapaxes(res, 0, 1)
    train_data = res
    ###
    # out = train_data[0,:,:,:]
    # out2 = out[out>0]
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    f = gzip.GzipFile('encoded_all.npy.gz', "r")
    image_data = np.load(f)
    l2norm = np.sqrt((image_data * image_data).sum(axis=1))
    image_data = image_data / l2norm.reshape(-1, 1)
    model = ConvAutoencoder3()
    model.load_state_dict(torch.load("models_normal_map/model120.pth"))
    model.to(device)
    model.eval()
    i = 0
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images = data
        # images_0 = images>0
        # mean = np.mean(images[images_0])
        # std = np.std(images[images_0])

        # clear the gradients of all optimized variables
        # images = images.permute(0, 3, 1, 2).float().to(device)
        images = images.permute(0, 3, 1, 2).float().to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)

        disp = images.cpu().detach().numpy()[0, :, :, :]
        disp = disp.swapaxes(0,2)
        disp = disp.swapaxes(0,1)
        plt.imshow(disp)
        plt.show()

        disp = outputs.cpu().detach().numpy()[0, :, :, :]
        disp = disp.swapaxes(0,2)
        disp = disp.swapaxes(0,1)
        plt.imshow(disp)
        plt.show()

        tmp = train_out[i, :, :, :]

        encoded_image = model.encode(images).detach().cpu().numpy()

        dot_prob = image_data.dot(encoded_image.reshape(-1, 1))
        idx_max = np.argmax(dot_prob)
        print(idx_max)

        num_by_part = 2562*36/4
        part = idx_max//num_by_part + 1
        print(part)
        idx = idx_max-(part-1)*num_by_part
        print(idx)
        f = gzip.GzipFile(f'part{int(part)}.npy.gz', "r")
        image_data = np.load(f)

        plt.imshow(image_data[int(idx), :, :, :])
        plt.show()

        # B = np.einsum('kij->ijk', tmp)

        # plt.imshow(tmp, cmap='gray')
        # plt.show()
        #
        # i+=1
        print("sleep")
        # time.sleep(5)



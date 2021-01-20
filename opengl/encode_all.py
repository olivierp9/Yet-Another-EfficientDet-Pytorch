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
    encoded_images = np.zeros((2562*36, 128))
    c = 0
    print(2562*36)
    for i in range(1, 5):
        f = gzip.GzipFile(f'part{i}.npy.gz', "r")
        image_data = np.load(f)

        train_loader = torch.utils.data.DataLoader(image_data, batch_size=1, num_workers=0, shuffle=False)

        model = ConvAutoencoder3()
        model.load_state_dict(torch.load("models_normal_map/model120.pth"))
        model.to(device)
        model.eval()

        for data in train_loader:
            images = data
            images = images.permute(0, 3, 1, 2).float().to(device)
            encoded_images[c, :] = model.encode(images).cpu().detach().numpy()
            c += 1
    print(c)
    f = gzip.GzipFile("encoded_all.npy.gz", "w")
    np.save(file=f, arr=encoded_images)
    f.close()
# affine
# crop+pad
# corsedropout
# Cutout
# gaussianblur
import time
import matplotlib.pyplot as plt
import numpy as np
from imgaug.augmenters import Sequential, Sometimes, CropAndPad, Multiply, Cutout, Affine, GaussianBlur

seq = Sequential([
    Sometimes(0.5, CropAndPad(percent=(-0.2, 0.2))),
    Sometimes(0.5, Cutout(nb_iterations=(1, 5),cval=[0,1], squared=False, size=0.2)),
    Sometimes(0.5, Affine(scale=(0.75, 1.5))),
    Sometimes(0.5, GaussianBlur(sigma=(0.0, 3.0))),
    Sometimes(0.5, Multiply((0.05, 1.0))),
], random_order=False)

test = np.load("../data/700mm_norm_1.npy")
test = np.array(test, dtype=np.float32)
test2 = seq(images=test)

for i in range(test2.shape[0]):
    tmp = test2[i, :, :, :]
    max = np.max(tmp)
    min = np.min(tmp)
    tmp[tmp!=0] += np.random.uniform(-min, 1-max)

np.save("../data/700mm_norm_1_aug.npy", test2)

for i in range(test2.shape[0]):
    plt.imshow(test2[i, :, :, :], cmap='gray',vmax=1, vmin=0)
    plt.show()
    plt.imshow(test2[i, :, :, :], cmap='gray')
    plt.show()
    time.sleep(1)

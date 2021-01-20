# affine
# crop+pad
# corsedropout
# Cutout
# gaussianblur
import time
import matplotlib.pyplot as plt
import numpy as np
from imgaug.augmenters import Sequential, Sometimes, CropAndPad, Multiply, Cutout, Affine, GaussianBlur
import gzip

seq = Sequential([
    Sometimes(0.5, Affine(scale=(0.5, 3.0))),
    Sometimes(0.5, CropAndPad(percent=(-0.2, 0.2))),
    Sometimes(0.5, Cutout(nb_iterations=(1, 5), cval=(0.0, 1.0), squared=False, size=0.2, fill_per_channel=True)),
    Sometimes(0.5, Cutout(nb_iterations=(1, 5), cval=(0.0, 1.0), squared=False, size=0.5, fill_per_channel=True,
                          fill_mode="background")),
    Sometimes(0.5, GaussianBlur(sigma=(0.0, 1.2))),
], random_order=False)

f = gzip.GzipFile('../my_array.npy.gz', "r")
img = np.load(f)

# test = np.load("../data/700mm_norm_1.npy")
test = np.array(img, dtype=np.float32)

test2 = seq(images=test)

# for i in range(test2.shape[0]):
#     tmp = test2[i, :, :, :]
#     max = np.max(tmp)
#     min = np.min(tmp)
#     tmp[tmp != 0] += np.random.uniform(-min, 1-max)
#
# np.save("../data/700mm_norm_1_aug.npy", test2)
f = gzip.GzipFile("../data/augz.npy.gz", "w")
np.save(file=f, arr=test2)
f.close()

#
# for i in range(test2.shape[0]):
#     plt.imshow(test2[i, :, :, :])
#     plt.show()
#     plt.imshow(test[i, :, :, :])
#     plt.show()
#     time.sleep(1)

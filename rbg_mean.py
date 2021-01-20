import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # base_path = "/home/olivier/Desktop/Yet-Another-EfficientDet-Pytorch/datasets/chess/train"


    # image_path = [s for s in os.listdir(base_path) if s.endswith('.jpg')]
    # ori_imgs = [cv2.imread(os.path.join(base_path, img_path))[..., ::-1] for img_path in image_path]
    # ori_imgs = np.array(ori_imgs)
    # ori_imgs = ori_imgs.reshape((ori_imgs.shape[0]*ori_imgs.shape[1]*ori_imgs.shape[2], 3))
    # ori_imgs_mean = np.mean(ori_imgs, axis=0)/255
    # ori_imgs_std = np.std(ori_imgs, axis=0)/255
    # test = 0

    # base_path = "/home/olivier/Desktop/Yet-Another-EfficientDet-Pytorch/datasets/bunny/train"
    base_path = "/home/olivier/Desktop/Yet-Another-EfficientDet-Pytorch/datasets/bunny/train"
    image_path = [s for s in os.listdir(base_path) if s.endswith('.npy')]
    arr = np.load(f"{base_path}/{image_path[0]}")

    plt.imshow(arr, cmap="gray", vmax=1, vmin=-1)
    plt.show()
    ori_imgs = [np.load(os.path.join(base_path, img_path))[..., ::-1] for img_path in image_path]
    ori_imgs = np.array(ori_imgs)
    ori_imgs_mean = np.mean(ori_imgs)
    ori_imgs_std = np.std(ori_imgs)
    print(np.max(ori_imgs))
    print(np.min(ori_imgs))
    print(ori_imgs_mean)
    print(ori_imgs_std)
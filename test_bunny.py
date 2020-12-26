import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

compound_coef = 0
force_input_size = None  # set None to use default size
img_path = 'datasets/bunny/val/0a02ad84-fb7d-4c08-a7f1-06391572f4ff.npy'

threshold = 0.15
iou_threshold = 0.0

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['bunny']
if __name__ == "__main__":
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    base_path = "datasets/bunny/val"
    list_of_img = os.listdir(base_path)
    list_of_img = [f"{base_path}/{name}" for name in list_of_img]
    ori_imgs, framed_imgs, framed_metas = preprocess(list_of_img, max_size=input_size, mean=[0.05], std=[0.16])

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                                 # replace this part with your project's anchor config
                                 ratios=[(1.0, 1.0), (0.7, 1.4), (1.4, 0.7)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0), 2 ** (4.0 / 3.0)])

    model.load_state_dict(torch.load('logs/bunny/efficientdet-d0_22_26000.pth'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue
        ori_imgs[i] = ori_imgs[i].copy()*255
        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 1)
            obj = obj_list[out[i]['class_ids'][j]]
            score = float(out[i]['scores'][j])
            print(score)
            cv2.putText(np.repeat(ori_imgs[i],3,2), '{:.3f}'.format(score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 0), 1)

            plt.imshow(ori_imgs[i])
    plt.show()


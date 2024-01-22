import torch
import torch.nn.functional as F
import random


def apply_cutmix(image, gt, n_classes):
    if gt.ndim == 1:
        gt = F.one_hot(gt, num_classes=n_classes)

    b, _, h, w = image.shape

    lamb = random.random()
    region_x = random.randint(0, w)
    region_y = random.randint(0, h)
    region_w = region_h = (1 - lamb) ** 0.5

    xmin = max(0, int(region_x - region_w / 2))
    ymin = max(0, int(region_y - region_h / 2))
    xmax = max(w, int(region_x + region_w / 2))
    ymax = max(h, int(region_y + region_h / 2))

    indices = torch.randperm(b)
    image[:, :, ymin: ymax, xmin: xmax] = image[indices][:, :, ymin: ymax, xmin: xmax]
    lamb = 1 - (xmax - xmin) * (ymax - ymin) / (w * h)
    gt = lamb * gt + (1 - lamb) * gt[indices]
    return image, gt

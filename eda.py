import sys

sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Kaggle-Iceberg/")

import torch
import albumentations as A
import torchvision.transforms.functional as TF

import config
from data import load_data, get_dls


if __name__ == "__main__":
    train_val_data = load_data("/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge/train.json")
    train_dl, val_dl = get_dls(
        train_val_data, val_ratio=config.VAL_RATIO, batch_size=config.BATCH_SIZE,
    )
    input_images = list()
    for input_image, gt in train_dl:
        input_images.append(input_image)
    all_input_image = torch.cat(input_images, dim=0)
    # mean_input_image = torch.mean(all_input_image, dim=0, keepdim=True)
    mean_input_image = torch.mean(all_input_image, dim=0)
    grid = TF.to_pil_image(mean_input_image)
    grid.show()

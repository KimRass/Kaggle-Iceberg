# Source: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

# "High winds will generate a brighter background. Conversely, low winds will generate a darker background."
# "The Sentinel-1 satellite is a side looking radar, which means it sees the image area at an angle
# (incidence angle). Generally, the ocean background will be darker at a higher incidence angle."
# "More advanced radars like Sentinel-1, can transmit and receive in the horizontal and vertical plane.
# Using this, you can get what is called a dual-polarization image."

import sys

sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Kaggle-Iceberg/")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import config
from utils import load_data, to_pil, save_image, image_to_grid
from data_aug import apply_cutmix

np.set_printoptions(linewidth=70)

IMG_SIZE = 75


def sample_to_img(sample):
    band1 = np.array(sample["band_1"]).reshape(IMG_SIZE, IMG_SIZE)
    band2 = np.array(sample["band_2"]).reshape(IMG_SIZE, IMG_SIZE)
    band3 = (band1 + band2) / 2
    # band3 = band1 / band2
    # band3 = np.divide(band1, band2, out=np.zeros_like(band1), where=(band2 != 0))

    # img = np.stack([band1, band2, band3], axis=2)
    img = np.stack([band3, band2, band1], axis=2)
    img = np.exp(img / 10)
    img = np.clip(img, 0, 1)
    img *= 255
    img = img.astype("uint8")
    return img


def get_mean_and_std(images):
    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for image in images:
        tensor = T.ToTensor()(image)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    return mean, std


def data_to_lists(data, training=False):
    imgs = list()
    inc_angles = list()
    if training:
        gts = list()
    else:
        ids = list()

    for sample in data:
        img = sample_to_img(sample)
        imgs.append(img)

        inc_angles.append(sample["inc_angle"])
        if training:
            gts.append(sample["is_iceberg"])
        else:
            ids.append(sample["id"])
    if training:
        return imgs, inc_angles, gts
    else:
        return ids, imgs, inc_angles
train_data, val_data = train_test_split(train_val_data, test_size=0.2)
train_imgs, train_inc_angles, train_gts = data_to_lists(train_data, training=True)
val_imgs, val_inc_angles, val_gts = data_to_lists(val_data, training=True)
test_ids, test_imgs, test_inc_angles = data_to_lists(test_data, training=False)

mean, std = get_mean_and_std(train_imgs)


class IcebergDataset(Dataset):
    def __init__(self, data, mean, std, training=True):
        super().__init__()

        self.training = training

        self.imgs = list()
        self.inc_angles = list()
        if training:
            self.gts = list()
        else:
            self.ids = list()

        for sample in data:
            img = sample_to_img(sample)
            self.imgs.append(img)

            self.inc_angles.append(sample["inc_angle"])
            if training:
                self.gts.append(sample["is_iceberg"])
            else:
                self.ids.append(sample["id"])

        if training:
            self.transformer = A.Compose(
                [
                    A.Flip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.3,
                        scale_limit=0,
                        rotate_limit=180,
                        border_mode=cv2.BORDER_WRAP,
                        p=1,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformer = A.Compose(
                [A.Normalize(mean=mean, std=std), ToTensorV2()]
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        augmented = self.transformer(image=img)
        image = augmented["image"]
        if self.training:
            gt = self.gts[idx]
            return image, gt
        else:
            id = self.ids[idx]
            return id, image


def get_dls(train_val_data, val_ratio, batch_size, n_cpus):
    train_val_ds = IcebergDataset(train_val_data)
    train_ds, val_ds = random_split(train_val_ds, lengths=(1 - val_ratio, val_ratio))

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
    )
    return train_dl, val_dl


def get_test_dl(test_data, batch_size, n_cpus):
    test_ds = IcebergDataset(test_data, training=False)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpus,
        pin_memory=False,
        drop_last=False,
    )
    return test_dl


if __name__ == "__main__":
    train_val_data = load_data("/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge/train.json")
    train_dl, val_dl = get_dls(
        train_val_data, val_ratio=config.VAL_RATIO, batch_size=config.BATCH_SIZE,
    )
    for image, gt in train_dl:
        # image, gt = apply_cutmix(image=image, gt=gt, n_classes=2)
        grid = image_to_grid(image, n_cols=4)
        grid.show()
        break

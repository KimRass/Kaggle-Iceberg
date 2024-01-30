# Source: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge
# References:
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md

# "High winds will generate a brighter background. Conversely, low winds will generate a darker background."
# "The Sentinel-1 satellite is a side looking radar, which means it sees the image area at an angle
# (incidence angle). Generally, the ocean background will be darker at a higher incidence angle."
# "More advanced radars like Sentinel-1, can transmit and receive in the horizontal and vertical plane.
# Using this, you can get what is called a dual-polarization image."

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import config
from utils import load_data

np.set_printoptions(linewidth=70)

IMG_SIZE = 75


def sample_to_img(sample):
    band1 = np.array(sample["band_1"]).reshape(IMG_SIZE, IMG_SIZE)
    band2 = np.array(sample["band_2"]).reshape(IMG_SIZE, IMG_SIZE)
    # band3 = (band1 + band2) / 2
    # band3 = band1 / band2
    # band3 = np.divide(band1, band2, out=np.zeros_like(band1), where=(band2 != 0))

    # img = np.stack([band3, band2, band1], axis=2)
    img = np.stack([band1, band2], axis=2)
    img = np.exp(img / 10)
    img = np.clip(img, 0, 1)
    img *= 255
    img = img.astype("uint8")
    return img


def get_mean_and_std(imgs):
    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for img in imgs:
        tensor = TF.to_tensor(img)
        
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

    for sample in data:
        img = sample_to_img(sample)
        imgs.append(img)

        inc_angles.append(sample["inc_angle"])
        if training:
            gts.append(sample["is_iceberg"])
    if training:
        return imgs, inc_angles, gts
    else:
        return imgs, inc_angles


class IcebergDataset(Dataset):
    def __init__(
        self,
        imgs,
        inc_angles,
        gts,
        img_mean,
        img_std,
        inc_angle_mean,
        inc_angle_std,
        training,
    ):
        super().__init__()

        self.imgs = imgs
        self.inc_angles = inc_angles
        self.gts = gts

        if training:
            self.transformer = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=0,
                        rotate_limit=180,
                        border_mode=cv2.BORDER_WRAP,
                        p=1,
                    ),
                    A.Normalize(mean=img_mean, std=img_std),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformer = A.Compose(
                [A.Normalize(mean=img_mean, std=img_std), ToTensorV2()]
            )

        self.inc_angles = (self.inc_angles - inc_angle_mean) / inc_angle_std

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        inc_angle = self.inc_angles[idx]
        augmented = self.transformer(image=img)
        image = augmented["image"]
        gt = self.gts[idx]
        return image, round(inc_angle, 4), gt


class IcebergDatasetForPrediction(Dataset):
    def __init__(
        self,
        imgs,
        inc_angles,
        img_mean,
        img_std,
        inc_angle_mean,
        inc_angle_std,
    ):
        super().__init__()

        self.imgs = imgs
        self.inc_angles = inc_angles

        self.transformer = A.Compose(
            [A.Normalize(mean=img_mean, std=img_std), ToTensorV2()]
        )

        self.inc_angles = (self.inc_angles - inc_angle_mean) / inc_angle_std

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        inc_angle = self.inc_angles[idx]
        augmented = self.transformer(image=img)
        image = augmented["image"]
        return image, round(inc_angle, 4)


def split_by_inc_angle(imgs, inc_angles, gts):
    train_imgs = list()
    train_inc_angles = list()
    train_gts = list()
    test_imgs = list()
    test_inc_angles = list()
    test_gts = list()
    for img, inc_angle, gt in zip(imgs, inc_angles, gts):
        if isinstance(inc_angle, float):
            train_imgs.append(img)
            train_inc_angles.append(inc_angle)
            train_gts.append(gt)
        else:
            test_imgs.append(img)
            test_inc_angles.append(inc_angle)
            test_gts.append(gt)
    return (
        train_imgs,
        train_inc_angles,
        train_gts,
        test_imgs,
        test_inc_angles,
        test_gts,
    )


def fill_missing_inc_angles(inc_angles, preds_path):
    inc_angle_preds = np.load(preds_path)
    
    new_inc_angles = np.array(inc_angles).copy()
    new_inc_angles[new_inc_angles == "na"] = inc_angle_preds
    return list(new_inc_angles.astype(np.float32))


def get_train_val_dls(train_json_path, inc_angle_preds_path, batch_size, n_cpus, seed):
    train_val_data = load_data(train_json_path)
    train_val_imgs, train_val_inc_angles, train_val_gts = data_to_lists(
        train_val_data, training=True,
    )
    train_val_inc_angles = fill_missing_inc_angles(
        inc_angles=train_val_inc_angles, preds_path=inc_angle_preds_path,
    )

    (
        train_imgs,
        val_imgs,
        train_inc_angles,
        val_inc_angles,
        train_gts,
        val_gts,
    ) = train_test_split(
        train_val_imgs,
        train_val_inc_angles,
        train_val_gts,
        test_size=config.VAL_RATIO,
        random_state=seed,
        shuffle=True,
    )

    img_mean, img_std = get_mean_and_std(train_imgs)
    train_ds = IcebergDataset(
        imgs=train_imgs,
        inc_angles=train_inc_angles,
        gts=train_gts,
        img_mean=img_mean,
        img_std=img_std,
        inc_angle_mean=np.array(train_inc_angles).mean(),
        inc_angle_std=np.array(train_inc_angles).std(),
        training=True,
    )
    val_ds = IcebergDataset(
        imgs=val_imgs,
        inc_angles=val_inc_angles,
        gts=val_gts,
        img_mean=img_mean,
        img_std=img_std,
        inc_angle_mean=np.array(train_inc_angles).mean(),
        inc_angle_std=np.array(train_inc_angles).std(),
        training=False,
    )

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


def get_test_dl(train_json_path, test_json_path, inc_angle_preds_path, batch_size, n_cpus):
    test_data = load_data(test_json_path)
    test_imgs, test_inc_angles = data_to_lists(test_data, training=False)

    train_val_data = load_data(train_json_path)
    train_val_imgs, train_val_inc_angles, _ = data_to_lists(
        train_val_data, training=True,
    )
    train_val_inc_angles = fill_missing_inc_angles(
        inc_angles=train_val_inc_angles, preds_path=inc_angle_preds_path,
    )

    train_imgs, _, train_inc_angles, _ = train_test_split(
        train_val_imgs, train_val_inc_angles, test_size=config.VAL_RATIO,
    )
    img_mean, img_std = get_mean_and_std(train_imgs)

    test_ds = IcebergDatasetForPrediction(
        imgs=test_imgs,
        inc_angles=test_inc_angles,
        img_mean=img_mean,
        img_std=img_std,
        inc_angle_mean=np.array(train_inc_angles).mean(),
        inc_angle_std=np.array(train_inc_angles).std(),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpus,
        pin_memory=False,
        drop_last=False,
    )

    ids = [sample["id"] for sample in test_data]
    return test_dl, ids


if __name__ == "__main__":
    train_dl, val_dl = get_train_val_dls(
        train_json_path="/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge/train.json",
        inc_angle_preds_path="/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge/inc_angle_pred.npy",
        batch_size=4,
        n_cpus=0,
    )

    img = cv2.imread("/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge/train_imgs/12.jpg")
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
    a = A.RandomResizedCrop(
        height=IMG_SIZE, width=IMG_SIZE, scale=(0.9, 1), ratio=(3 / 4, 4 / 3),
    )
    for _ in range(10):
        to_pil(a(image=img)["image"]).show()

    x = list(range(100))
    y = [0, 1, 2, 3] * 25
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds = kfold.split(x, y)
    for train_indices, val_indices in folds:
        np.array(x)[a]
        np.array(y)[b]
        break
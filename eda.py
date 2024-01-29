import sys

sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Kaggle-Iceberg/")

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import pandas as pd

import config
from utils import set_seed
from data import load_data, data_to_lists, split_by_inc_angle


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge",
    )

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def save_inc_angle_stripplot(inc_angles, gts, save_path):
    plt.figure(figsize=(8, 30))
    sns.stripplot(y=inc_angles, hue=gts, size=5, dodge=False)
    plt.tight_layout()
    plt.savefig(save_path)


train_val_data = load_data("/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge/train.json")
train_val_imgs, train_val_inc_angles, train_val_gts = data_to_lists(
    train_val_data, training=True,
)
train_val_df = pd.DataFrame(train_val_gts, columns=["gt"])
train_val_df.value_counts()


if __name__ == "__main__":
    set_seed(config.SEED)
    args = get_args()

    train_val_data = load_data(Path(args.DATA_DIR)/"train.json")
    train_val_imgs, train_val_inc_angles, train_val_gts = data_to_lists(train_val_data, training=True)
    (
        train_imgs,
        train_inc_angles,
        train_gts,
        test_imgs,
        test_inc_angles,
        test_gts,
    ) = split_by_inc_angle(
        imgs=train_val_imgs, inc_angles=train_val_inc_angles, gts=train_val_gts,
    )

    save_inc_angle_stripplot(
        inc_angles=train_inc_angles,
        gts=train_gts,
        save_path=Path(__file__).resolve().parent/"resources/inc_angle.jpg",
    )

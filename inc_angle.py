from sklearn.ensemble import RandomForestRegressor
import numpy as np
import argparse
from pathlib import Path

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

    model = RandomForestRegressor(random_state=config.SEED)
    model.fit(np.array(train_gts)[:, None], np.array(train_inc_angles))

    preds = model.predict(np.array(test_gts)[:, None])
    np.save(Path(args.DATA_DIR)/"inc_angle_pred.npy", preds)

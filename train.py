import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import math
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import argparse

import config
from utils import set_seed, load_data, to_pil, save_image, get_device
from data import get_dls
from model import Classifier
from data_aug import apply_cutmix


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123, required=False)
    parser.add_argument("--n_epochs", type=int, default=60, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    # parser.add_argument("--lr", type=float, default=0.0005, required=False)
    # parser.add_argument("--data_dir", type=str, required=True)
    # parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def train_single_step(input_image, gt, model, optim, device):
    input_image = input_image.to(device)
    gt = gt.to(device)
    image, gt = apply_cutmix(image=input_image, gt=gt, n_classes=2)

    loss = model.get_loss(input_image=input_image, gt=gt)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


@torch.no_grad()
def validate(dl, model, device):
    model.eval()

    cum_loss = 0
    for input_image, gt in dl:
        input_image = input_image.to(device)
        gt = gt.to(device)

        loss = model.get_loss(input_image=input_image, gt=gt)
        cum_loss += loss.item()

    model.train()
    return cum_loss / len(dl)


def train(n_epochs, train_dl, val_dl, model, optim, device):
    best_val_loss = math.inf
    save_dir = "/Users/jongbeomkim/Desktop/workspace/Kaggle-Iceberg/model_params"
    prev_save_path = Path(".pth")
    for epoch in range(1, n_epochs + 1):
        cum_loss = 0
        for input_image, gt in tqdm(train_dl, leave=False):
            loss = train_single_step(
                input_image=input_image, gt=gt, model=model, optim=optim, device=device,
            )
            cum_loss += loss.item()
        train_loss = cum_loss / len(train_dl)

        log = f"""[ {epoch}/{n_epochs} ]"""
        log += f"[ Train loss: {train_loss:.4f} ]"

        val_loss = validate(dl=val_dl, model=model, device=device)
        log += f"[ Val loss: {val_loss:.4f} ]"
        print(log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cur_save_path = str(Path(save_dir)/f"epoch_{epoch}-val_loss_{val_loss:.4f}.pth")
            torch.save(model.state_dict(), cur_save_path)
            if prev_save_path.exists():
                prev_save_path.unlink()
            prev_save_path = Path(cur_save_path)


def main():
    args = get_args()
    set_seed(args.SEED)
    # DEVICE = get_device()
    DEVICE = torch.device("cpu")

    train_val_data = load_data("/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge/train.json")
    train_dl, val_dl = get_dls(
        train_val_data,
        val_ratio=config.VAL_RATIO,
        batch_size=args.BATCH_SIZE,
        n_cpus=0,
    )

    model = Classifier().to(DEVICE)
    optim = AdamW(model.parameters(), lr=0.0001)

    train(
        n_epochs=args.N_EPOCHS,
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optim=optim,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()

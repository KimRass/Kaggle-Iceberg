import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import argparse

import config
from utils import set_seed, load_data, to_pil, save_image, get_device
from data import get_test_dl
from model import Classifier


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
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


@torch.no_grad()
def predict(test_dl, model, device):
    model.eval()

    ids = list()
    preds = list()
    for id, input_image in tqdm(test_dl):
        ids.extend(id)

        input_image = input_image.to(device)
        pred = model(input_image.detach())
        pred = pred.cpu()
        softmax = F.softmax(pred, dim=1)
        preds.extend(list(softmax[:, 1].numpy()))
        # break
    pred_df = pd.DataFrame({"id": ids, "is_iceberg": preds})

    model.train()
    return pred_df


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = torch.device("cpu")

    test_data = load_data("/Users/jongbeomkim/Documents/datasets/statoil-iceberg-classifier-challenge/test.json")
    test_dl = get_test_dl(test_data, batch_size=args.BATCH_SIZE, n_cpus=0)

    model = Classifier().to(DEVICE)
    # state_dict = torch.load(args.MODEL_PARAMS)
    state_dict = torch.load("/Users/jongbeomkim/Desktop/workspace/Kaggle-Iceberg/model_params/epoch_53-val_loss_0.2597.pth")
    model.load_state_dict(state_dict)

    pred_df = predict(test_dl=test_dl, model=model, device=DEVICE)
    pred_df.to_csv("/Users/jongbeomkim/Desktop/workspace/Kaggle-Iceberg/submission/submission.csv", index=False)


if __name__ == "__main__":
    main()
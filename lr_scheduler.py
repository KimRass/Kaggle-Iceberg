# References:
    # https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt

from utils import plt_to_pil, show_image


def get_lrs(optim, scheduler, n_epochs):
    lrs = list()
    for epoch in range(n_epochs):
        lr = optim.param_groups[0]["lr"]
        lrs.append(lr)
        scheduler.step(epoch=epoch + 1)
    return lrs


def vis_lrs(lrs, n_epochs):
    fig, axes = plt.subplots(figsize=(8, 3))
    axes.plot(range(1, n_epochs + 1), lrs, linewidth=1)
    axes.set_ylim([0, max(lrs) * 1.1])
    axes.set_xlim([1, n_epochs])
    axes.tick_params(axis="x", labelrotation=90, labelsize=5)
    axes.tick_params(axis="y", labelsize=5)
    axes.grid(axis="x", color="black", alpha=0.5, linestyle="--", linewidth=0.5)
    fig.tight_layout()

    arr = plt_to_pil(fig)
    return arr


def get_cosine_scheduler(optim, warmup_epochs, n_epochs, init_lr):
    return CosineLRScheduler(
        optimizer=optim,
        t_initial=n_epochs,
        warmup_t=warmup_epochs,
        warmup_lr_init=init_lr,
        warmup_prefix=True,
        t_in_epochs=True,
    )


if __name__ == "__main__":
    model = torch.nn.Linear(2, 1)

    n_epochs = 80
    # lr = 0.0005
    lr = 10
    optim = AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_scheduler(
        optim=optim, warmup_epochs=10, n_epochs=n_epochs, init_lr=lr * 0.2,
    )

    lrs = get_lrs(optim=optim, scheduler=scheduler, n_epochs=n_epochs)
    vis = vis_lrs(lrs=lrs, n_epochs=n_epochs)
    show_image(vis)

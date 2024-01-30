# Reference:
    # https://stackoverflow.com/questions/47818968/adding-an-additional-value-to-a-convolutional-neural-network-input
    # https://github.com/aadhithya/AdaIN-pytorch/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import ssl

from utils import print_n_prams

ssl._create_default_https_context = ssl._create_unverified_context


class AdaIN(nn.Module):
    # "Unlike BN, IN or CIN, AdaIN has no learnable affine parameters. Instead, it adaptively
    # computes the affine parameters from the style input."
    def __init__(self, eps=1e-5):
        super().__init__()

        self.eps = eps

    def _get_mean_and_std(self, x):
        x = torch.flatten(x, start_dim=2, end_dim=3)
        # "Similar to IN, these statistics are computed across spatial locations."
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps) ** 0.5
        return mean[..., None], std[..., None]

    def forward(self, cont_image, sty_image):
        # "$\text{AdaIN}(x, y) = \sigma(y)\bigg(\frac{x - \mu(x)}{\sigma(x)}\bigg) + \mu(y)$"
        # "AdaIN receives a content input $x$ and a style input $y$, and simply aligns the channel-wise
        # mean and variance of $x$ to match those of $y$."
        cont_mean, cont_std = self._get_mean_and_std(cont_image)
        sty_mean, sty_std = self._get_mean_and_std(sty_image)
        # "The output produced by AdaIN will preserve the spatial structure of the content image."
        x = (cont_image - cont_mean) / cont_std
        x = x * sty_std + sty_mean
        return x


class CELossWithLabelSmoothing(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

    def forward(self, pred, gt, label_smoothing=0):
        assert 0 <= label_smoothing <= 1, "The argument `label_smoothing` must be between 0 and 1!"

        if gt.ndim == 1:
            gt = torch.eye(self.n_classes, device=gt.device)[gt]
            return self(pred, gt, label_smoothing=label_smoothing)
        elif gt.ndim == 2:
            log_prob = F.log_softmax(pred, dim=1)
            ce_loss = -torch.sum(gt * log_prob, dim=1)
            loss = (1 - label_smoothing) * ce_loss
            loss += label_smoothing * -torch.sum(log_prob, dim=1)
            return torch.mean(loss)


class ResNet50BasedClassifier(nn.Module):
    def __init__(self, pretrained=True, adain=False):
        super().__init__()

        self.adain = adain

        if pretrained:
            self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.resnet50 = resnet50()

        if adain:
            self.norm = AdaIN()
            self.resnet50.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        self.resnet50.fc = nn.Linear(2048, 2)

        self.loss_fn = CELossWithLabelSmoothing(n_classes=2)

    def forward(self, image, inc_angle):
        if self.adain:
            sty_image = inc_angle[:, None, None, None]
            x = self.norm(cont_image=image, sty_image=sty_image)
        else:
            _, _, img_size, _ = image.shape
            expanded_inc_angle = inc_angle[:, None, None, None].repeat(1, 1, img_size, img_size)
            x = torch.cat([image, expanded_inc_angle], dim=1)

        x = self.resnet50(x)
        return x

    def get_loss(self, image, inc_angle, gt, label_smoothing):
        pred = self(image=image, inc_angle=inc_angle)
        return self.loss_fn(pred, gt, label_smoothing=label_smoothing)

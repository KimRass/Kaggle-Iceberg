# Reference:
    # https://stackoverflow.com/questions/47818968/adding-an-additional-value-to-a-convolutional-neural-network-input

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.vgg.features[0] = nn.Conv2d(3 + 1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.classifier[6] = nn.Linear(4096, 2)

    def forward(self, image, inc_angle):
        # image = torch.randn(2, 3, 75, 75)
        # inc_angle = torch.randn(2)

        _, _, img_size, _ = image.shape
        expanded_inc_angle = inc_angle[:, None, None, None].repeat(1, 1, img_size, img_size)
        x = torch.cat([image, expanded_inc_angle], dim=1)
        x = self.vgg(x)
        return x

    def get_loss(self, image, inc_angle, gt):
        pred = self(image=image, inc_angle=inc_angle)
        return F.cross_entropy(input=pred, target=gt, reduction="mean")


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.feat_ext = vgg.features
        self.avg_pool = vgg.avgpool
        self.cls_head = nn.Sequential(
            *[
                nn.Linear(25088 + 1, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 2),
            ]
        )

    def forward(self, image, inc_angle):
        # image = torch.randn(2, 3, 75, 75)
        # inc_angle = torch.randn(2)

        x = self.feat_ext(image)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = torch.cat([x, inc_angle.unsqueeze(1)], dim=1)
        x = self.cls_head(x)
        return x

    def get_loss(self, image, inc_angle, gt):
        pred = self(image=image, inc_angle=inc_angle)
        return F.cross_entropy(input=pred, target=gt, reduction="mean")


if __name__ == "__main__":
    model = Classifier()
    image = torch.randn(32, 3, 75, 75)
    inc_angle = torch.randn(32, 1)
    model(image, inc_angle)

    vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    vgg.classifier[6] = nn.Linear(4096, 2)
    vgg.features[0] = nn.Conv2d(3 + 1, 64, kernel_size=3, stride=1, padding=1)
    x = torch.randn(2, 4, 75, 75)

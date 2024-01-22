import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import ssl

from utils import load_data, to_pil, save_image
from data import IcebergDataset

ssl._create_default_https_context = ssl._create_unverified_context


class IncAnglePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.layers.classifier[6] = nn.Linear(4096, 1)

    def forward(self, x):
        return self.layers(x)

    def get_loss(self, input_image, gt):
        pred = self(input_image)
        return F.mse_loss(input=pred, target=gt, reduction="mean")

    # def get_acc(self, input_image, gt):
    #     pred = self(input_image)
    #     argmax = torch.argmax(pred, dim=1)
    #     corr = (argmax == gt).float()
    #     acc = corr.mean()
    #     return acc


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg16_bn()
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

    def forward(self, x, inc_angle):
        x = self.feat_ext(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = torch.cat([x, inc_angle], dim=1)
        x = self.cls_head(x)
        return x

    def get_loss(self, input_image, gt):
        pred = self(input_image)
        return F.cross_entropy(input=pred, target=gt, reduction="mean")

    def get_acc(self, input_image, gt):
        pred = self(input_image)
        argmax = torch.argmax(pred, dim=1)
        corr = (argmax == gt).float()
        acc = corr.mean()
        return acc


if __name__ == "__main__":
    model = Classifier()
    input_image = torch.randn(32, 3, 75, 75)
    inc_angle = torch.randn(32, 1)
    model(input_image, inc_angle)

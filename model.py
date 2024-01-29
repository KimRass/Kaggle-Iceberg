# Reference:
    # https://stackoverflow.com/questions/47818968/adding-an-additional-value-to-a-convolutional-neural-network-input

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import resnet50, ResNet50_Weights
import ssl

from utils import print_n_prams

ssl._create_default_https_context = ssl._create_unverified_context


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
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.resnet50 = resnet50()
        # self.resnet50.conv1 = nn.Conv(2, 64, kernel_size=7, stride=2, padding=3)
        self.resnet50.fc = nn.Linear(2048, 2)

        self.loss_fn = CELossWithLabelSmoothing(n_classes=2)

    def forward(self, image, inc_angle):
        _, _, img_size, _ = image.shape
        expanded_inc_angle = inc_angle[:, None, None, None].repeat(1, 1, img_size, img_size)
        x = torch.cat([image, expanded_inc_angle], dim=1)

        x = self.resnet50(x)
        return x

    def get_loss(self, image, inc_angle, gt, label_smoothing):
        pred = self(image=image, inc_angle=inc_angle)
        return self.loss_fn(pred, gt, label_smoothing=label_smoothing)


# class Classifier(nn.Module):
#     def __init__(self, pretrained=False):
#         super().__init__()

#         if pretrained:
#             self.vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
#         else:
#             self.vgg = vgg16_bn()
#         self.vgg.features[0] = nn.Conv2d(3 + 1, 64, kernel_size=3, stride=1, padding=1)
#         self.cls_head = nn.Sequential(
#             *[
#                 nn.Linear(2048, 512),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5),
#                 nn.Linear(512, 512),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5),
#                 nn.Linear(512, 2),
#             ]
#         )

#     def forward(self, image, inc_angle):
#         _, _, img_size, _ = image.shape
#         expanded_inc_angle = inc_angle[:, None, None, None].repeat(1, 1, img_size, img_size)
#         x = torch.cat([image, expanded_inc_angle], dim=1)

#         x = self.vgg.features(x)
#         x = torch.flatten(x, start_dim=1, end_dim=3)
#         x = self.cls_head(x)
#         return x

#     def get_loss(self, image, inc_angle, gt, label_smoothing):
#         pred = self(image=image, inc_angle=inc_angle)
#         return F.cross_entropy(
#             input=pred, target=gt, reduction="mean", label_smoothing=label_smoothing,
#         )


# class Classifier(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()

#         if pretrained:
#             self.vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
#         else:
#             self.vgg = vgg16_bn()
#         self.vgg.features[0] = nn.Conv2d(3 + 1, 64, kernel_size=3, stride=1, padding=1)
#         self.vgg.classifier[0] = nn.Linear(512 * 2 * 2, 4096)
#         self.vgg.classifier[6] = nn.Linear(4096, 2)

#     def forward(self, image, inc_angle):
#         # image = torch.randn(2, 3, 75, 75)
#         # inc_angle = torch.randn(2)

#         _, _, img_size, _ = image.shape
#         expanded_inc_angle = inc_angle[:, None, None, None].repeat(1, 1, img_size, img_size)
#         x = torch.cat([image, expanded_inc_angle], dim=1)

#         x = self.vgg.features(x)
#         x = torch.flatten(x, start_dim=1, end_dim=3)
#         x = self.vgg.classifier(x)
#         return x

#     def get_loss(self, image, inc_angle, gt, label_smoothing):
#         pred = self(image=image, inc_angle=inc_angle)
#         return F.cross_entropy(
#             input=pred, target=gt, reduction="mean", label_smoothing=label_smoothing,
#         )


# class Classifier(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()

#         if pretrained:
#             self.vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
#         else:
#             self.vgg = vgg16_bn()
#         self.vgg.features[0] = nn.Conv2d(3 + 1, 64, kernel_size=3, stride=1, padding=1)
#         self.vgg.classifier[6] = nn.Linear(4096, 2)

#     def forward(self, image, inc_angle):
#         _, _, img_size, _ = image.shape
#         expanded_inc_angle = inc_angle[:, None, None, None].repeat(1, 1, img_size, img_size)
#         x = torch.cat([image, expanded_inc_angle], dim=1)

#         x = self.vgg.features(x)

#         ori_device = x.device
#         x = x.to(torch.device("cpu"))
#         x = self.vgg.avgpool(x)
#         x = x.to(ori_device)

#         x = self.vgg.classifier(x)
#         return x

#     def get_loss(self, image, inc_angle, gt, label_smoothing):
#         pred = self(image=image, inc_angle=inc_angle)
#         return F.cross_entropy(
#             input=pred, target=gt, reduction="mean", label_smoothing=label_smoothing,
#         )


# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()

#         vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
#         self.feat_ext = vgg.features
#         self.avg_pool = vgg.avgpool
#         self.cls_head = nn.Sequential(
#             *[
#                 nn.Linear(25088 + 1, 4096),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5),
#                 nn.Linear(4096, 4096),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5),
#                 nn.Linear(4096, 2),
#             ]
#         )

#     def forward(self, image, inc_angle):
#         # image = torch.randn(2, 3, 75, 75)
#         # inc_angle = torch.randn(2)

#         x = self.feat_ext(image)
#         x = self.avg_pool(x)
#         x = torch.flatten(x, start_dim=1, end_dim=3)
#         x = torch.cat([x, inc_angle.unsqueeze(1)], dim=1)
#         x = self.cls_head(x)
#         return x


if __name__ == "__main__":
    model = Classifier()
    print_n_prams(model)
    print_n_prams(vgg16_bn())
    print_n_prams(vgg11_bn())
    print_n_prams(resnet50()) 
    
    model = resnet50()
    model

    image = torch.randn(32, 3, 75, 75)
    model.conv1 = nn.Conv(2, 64, kernel_size=7, stride=2, padding=3)
    model.fc = nn.Linear(2048, 2)
    
    out = model(image)
    out.shape
    
    inc_angle = torch.randn(32, 1)
    model(image, inc_angle)

    vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    vgg.classifier[6] = nn.Linear(4096, 2)
    vgg.features[0] = nn.Conv2d(3 + 1, 64, kernel_size=3, stride=1, padding=1)
    x = torch.randn(2, 4, 75, 75)


    pred = torch.randn(4, 2)
    gt = torch.randn(4, 2)
    log_prob = F.log_softmax(pred, dim=1)
    log_softmax
    F.ce_loss(log_softmax, gt)
    
    
    pred = torch.randn(4, 3)
    # gt = F.softmax(torch.randn(4, 3), dim=1)
    gt = torch.argmax(gt, dim=1)

    label_smoothing = 0
    n_classes = 3
    CELossWithLabelSmoothing(n_classes=n_classes)(pred, gt, label_smoothing=label_smoothing)
    ClassificationLoss(n_classes=n_classes)(pred, gt, label_smoothing=label_smoothing)
    # F.cross_entropy(pred, gt)

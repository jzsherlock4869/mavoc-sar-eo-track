import sys
sys.path.append('../')

import torch
from torch import nn as nn

import timm


class SimpleDualstreamArch(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=10):
        super().__init__()
        self.backbone_sar = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        self.backbone_eo = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        sar_in = self.backbone_sar.get_classifier().in_features
        eo_in = self.backbone_eo.get_classifier().in_features
        self.gap = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten()
        )
        self.classifier = nn.Linear(in_features=sar_in + eo_in, out_features=num_classes, bias=True)

    def forward(self, sar_img, eo_img):
        sar_feat = self.backbone_sar.forward_features(sar_img)
        eo_feat = self.backbone_eo.forward_features(eo_img)
        sar_vec, eo_vec = self.gap(sar_feat), self.gap(eo_feat)
        cat_vec = torch.cat((sar_vec, eo_vec), dim=1)
        out = self.classifier(cat_vec)
        return out



class ConcatInputArch(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=10):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        feat_in = self.backbone.get_classifier().in_features
        self.gap = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten()
        )
        self.classifier = nn.Linear(in_features=feat_in, out_features=num_classes, bias=True)

    def forward(self, sar_img, eo_img):
        x = torch.cat((eo_img[:, :2, :, :], sar_img[:, :1, :, :]), dim=1)
        feat = self.backbone.forward_features(x)
        feat_vec = self.gap(feat)
        out = self.classifier(feat_vec)
        return out


class AddDualstreamArch(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=10):
        super().__init__()
        self.backbone_sar = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        self.backbone_eo = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        sar_in = self.backbone_sar.get_classifier().in_features
        eo_in = self.backbone_eo.get_classifier().in_features
        assert sar_in == eo_in
        self.gap = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten()
        )
        self.classifier = nn.Linear(in_features=sar_in, out_features=num_classes, bias=True)

    def forward(self, sar_img, eo_img):
        sar_feat = self.backbone_sar.forward_features(sar_img)
        eo_feat = self.backbone_eo.forward_features(eo_img)
        sar_vec, eo_vec = self.gap(sar_feat), self.gap(eo_feat)
        # cat_vec = torch.cat((sar_vec, eo_vec), dim=1)
        fuse_vec = (sar_vec + eo_vec) / 2
        out = self.classifier(fuse_vec)
        return out
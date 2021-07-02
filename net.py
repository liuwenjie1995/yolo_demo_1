#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 下午5:40
# @Author  : wenjie.Liu
# @Site    :
# @File    : net.py
# @Software: PyCharm
import fastai.vision.models
import torch
import torchvision
from torch import nn
from fastai.vision.all import *
from collections import OrderedDict


def get_backbone(device):
    # vgg = torchvision.models.vgg16_bn(pretrained=True)
    # backbone = torch.nn.Sequential(*(list(vgg.children())[:-1]))
    # backbone.add_module('1',
    #                     nn.Sequential(
    #                         nn.AdaptiveAvgPool2d(output_size=(7, 7)),
    #                         nn.Conv2d(512, 1024, 1)
    #                     ))
    # debug
    # print(backbone)
    backbone = torch.nn.Sequential(
        OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3)),
            ('norm1', nn.BatchNorm2d(64)),
            ('pool1', nn.MaxPool2d(2, stride=2)),

            ('conv2', nn.Conv2d(64, 192, 3, 1, 1)),
            ('norm2', nn.BatchNorm2d(192)),
            ('pool2', nn.MaxPool2d(2, 2)),

            ('conv3_1', nn.Conv2d(192, 128, 1)),
            ('conv3_2', nn.Conv2d(128, 256, 3, 1, 1)),
            ('norm3_2', nn.BatchNorm2d(256)),
            ('conv3_3', nn.Conv2d(256, 256, 1)),
            ('conv3_4', nn.Conv2d(256, 512, 3, 1, 1)),
            ('norm3_4', nn.BatchNorm2d(512)),
            ('pool3', nn.MaxPool2d(2, 2)),

            ('conv4_1_1', nn.Conv2d(512, 256, 1)),
            ('conv4_1_2', nn.Conv2d(256, 512, 3, 1, 1)),
            ('norm4_1', nn.BatchNorm2d(512)),
            ('conv4_2_1', nn.Conv2d(512, 256, 1)),
            ('conv4_2_2', nn.Conv2d(256, 512, 3, 1, 1)),
            ('norm4_2', nn.BatchNorm2d(512)),
            ('conv4_3_1', nn.Conv2d(512, 256, 1)),
            ('conv4_3_2', nn.Conv2d(256, 512, 3, 1, 1)),
            ('norm4_4', nn.BatchNorm2d(512)),
            ('conv4_4_1', nn.Conv2d(512, 256, 1)),
            ('conv4_4_2', nn.Conv2d(256, 512, 3, 1, 1)),
            ('norm4_4', nn.BatchNorm2d(512)),

            ('conv4_5', nn.Conv2d(512, 1024, 3, 1, 1)),
            ('pool_4', nn.MaxPool2d(2, 2)),


            ('conv5_1_1', nn.Conv2d(1024, 512, 1)),
            ('conv5_1_2', nn.Conv2d(512, 1024, 3, 1, 1)),
            ('norm5_1', nn.BatchNorm2d(1024)),
            ('conv5_2_1', nn.Conv2d(1024, 512, 1)),
            ('conv5_2_2', nn.Conv2d(512, 1024, 3, 1, 1)),
            ('norm5_2', nn.BatchNorm2d(1024)),
            ('conv5_3', nn.Conv2d(1024, 1024, 3, 1, 1)),
            ('norm5_3', nn.BatchNorm2d(1024)),
            ('pool5', nn.MaxPool2d(2, 2)),

            ('conv5_1_1', nn.Conv2d(1024, 1024, 3, 1, 1)),
            ('norm5_1_1', nn.BatchNorm2d(1024)),
            ('conv5_1_2', nn.Conv2d(1024, 1024, 3, 1, 1)),
            ('norm5_1_2', nn.BatchNorm2d(1024)),
        ])
    )
    return backbone


class Yolo_v1(nn.Module):
    def __init__(self):
        super(Yolo_v1, self).__init__()
        self.backbone = get_backbone(device='cpu')
        # self.head = create_head(7*7*1024, 7*7*30)

        self.head = nn.Sequential(
            torch.nn.Dropout(.3),
            torch.nn.Linear(7*7*1024, 7*7*30)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(1, -1)
        x = self.head(x)
        x = x.view(1, 49, 30)
        return x


# debug
# yolo = Yolo_v1()
#
# img = torch.zeros([1, 3, 448, 448])
# out = yolo(img)
# print(out.shape)

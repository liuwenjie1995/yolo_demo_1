#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 下午1:57
# @Author  : wenjie.Liu
# @Site    :
# @File    : generate_data.py
# @Software: PyCharm
import os.path

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.voc import VOCDetection
from myutil import *

root = '/home/liu/work/data/YOLO_VOC'
TRAIN_DS = VOCDetection(root, image_set='train')
VALUE_DS = VOCDetection(root, image_set='val')

transform = transforms.Compose(
    [
        transforms.GaussianBlur,
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]
)


class Train_DS(Dataset):
    def __init__(self):
        super(Train_DS, self).__init__()
        self.img_root = r'/home/liu/work/data/YOLO_VOC/VOCdevkit/VOC2012/JPEGImages'

    def __len__(self):
        return len(TRAIN_DS)

    def __getitem__(self, i, dsize_w=512, dsize_h=512):
        img, json = TRAIN_DS[i]
        img = np.asanyarray(img)
        h, w, _ = img.shape
        img = cv2.resize(img, (512, 512))
        scale_w = 512 / w
        scale_h = 512 / h
        bbox2lbls = []
        for object in json['annotation']['object']:
            cls_name = object['name']
            bbox = (float(object['bndbox']['xmin']) * scale_w,
                    float(object['bndbox']['ymin']) * scale_h,
                    float(object['bndbox']['xmax']) * scale_w,
                    float(object['bndbox']['ymax']) * scale_h)
            bbox = [int(i) for i in bbox]
            bbox2lbls.append([bbox, cls_name])
        target = bboxes_lbl2target(bbox2lbls)
        img = transforms.ToTensor()(img)
        return img, target


class Value_DS(Dataset):
    def __init__(self):
        super(Value_DS, self).__init__()
        self.img_root = r'/home/liu/work/data/YOLO_VOC/VOCdevkit/VOC2012/JPEGImages'

    def __len__(self):
        return len(VALUE_DS)

    def __getitem__(self, i, dsize_w=512, dsize_h=512):
        img, json = VALUE_DS[i]
        img = np.asanyarray(img)
        h, w, _ = img.shape
        img = cv2.resize(img, (512, 512))

        scale_w = 512 / w
        scale_h = 512 / h
        bbox2lbls = []
        for object in json['annotation']['object']:
            cls_name = object['name']
            bbox = (float(object['bndbox']['xmin']) * scale_w,
                    float(object['bndbox']['ymin']) * scale_h,
                    float(object['bndbox']['xmax']) * scale_w,
                    float(object['bndbox']['ymax']) * scale_h)
            bbox = [int(i) for i in bbox]
            bbox2lbls.append([bbox, cls_name])
        target = bboxes_lbl2target(bbox2lbls)
        return img, target


train_ds, value_ds = Train_DS(), Value_DS()
train_loader, value_loader = DataLoader(train_ds,  shuffle=True, batch_size=1),\
                             DataLoader(value_ds,  shuffle=True, batch_size=1)

for imgs, lbls in train_loader:
    print(imgs.shape, lbls.shape)

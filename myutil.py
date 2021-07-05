#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 上午9:48
# @Author  : wenjie.Liu
# @Site    :
# @File    : utils.py.py
# @Software: PyCharm

import cv2
import pandas as pd
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Anchor:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


cls_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor']


def index2whctr(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    center_x = x1 + w / 2
    center_y = y1 + h / 2
    return w, h, center_x, center_y


def whctr2index(w, h, center_x, center_y):
    x1 = center_x - w / 2
    y1 = center_y - h / 2
    x2 = center_x + w / 2
    y2 = center_y + h / 2
    return x1, x2, y1, y2


def get_scaled_index(bbox, scale_h, scale_w):
    return (bbox[0] * scale_w,
            bbox[1] * scale_h,
            bbox[2] * scale_w,
            bbox[3] * scale_h)


def get_rescale_index(bbox, scale_h, scale_w):
    scale_w = 1 / float(scale_w)
    scale_h = 1 / float(scale_h)
    return (bbox[0] * scale_w,
            bbox[1] * scale_h,
            bbox[2] * scale_w,
            bbox[3] * scale_h)


def get_yolo_bbox(x1, y1, x2, y2, img_w=448, img_h=448, split=7.):
    # 分成49个格子，第(i, j)个格子代表有东西
    item_w = img_w / split
    item_h = img_h / split
    w, h, center_x, center_y = index2whctr(x1, y1, x2, y2)
    i = int(center_x / item_w)
    j = int(center_y / item_h)

    center_x = (center_x % item_w) / item_w
    center_y = (center_y % item_h) / item_h

    w = w / img_w
    h = h / img_h
    index = ((i - 1) * 7 + j) - 1
    return w, h, center_x, center_y, index


def name2cls(cls_name, cls_list: list, one_hot=True):
    cls_nums = len(cls_list)
    cls = cls_list.index(cls_name)

    if one_hot:
        lbl = np.zeros([cls_nums])
        lbl[cls] = 1
        return lbl
    else:
        lbl = cls
        return cls


def bboxes_lbl2target(yolo_bboxes):
    label = np.zeros([49, 30])
    # 每个格子可以预测两个框框，如果一个框框有两个目标那么就会丢失其中一个目标，毕竟都是只能预测一个目标的

    for bbox, cls_name in yolo_bboxes:
        #  7 * 7 * (2 * (w, h, center_x, center_y, fitness) * 30)
        cls = name2cls(cls_name, cls_list, one_hot=True)
        w, h, center_x, center_y, index = get_yolo_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
        bbox_info = np.asarray([w, h, center_x, center_y, 1])
        label[index] = np.hstack([bbox_info, bbox_info, cls])

    return label


def target2bboxes_lbles(target):
    yolo_bboxes, lbles, indexes = [], [], []
    for i in range(7 * 7):
        item = target[i]
        if sum(item) is not 0:
            bbox1 = (i for i in item[:5])
            bbox2 = (i for i in item[5:10])
            lbl = np.argmax(item[10:])

            indexes.append(i)
            yolo_bboxes.append([bbox1, bbox2])
            lbles.append(lbl)
    return yolo_bboxes, lbles, indexes


def yolo2whctr(output, split=7, img_w=448, img_h=448, ):
    # 先回复到448*448状态，在反scale
    item = 448 / split
    real_bboxes = []
    yolo_bboxes, lbles, indexes = target2bboxes_lbles(output)
    for bbox, index in zip(yolo_bboxes, indexes):
        i = index / split
        j = index % split

        center_x = i * item + bbox[0]
        center_y = j * item + bbox[1]
        w = img_w * bbox[2]
        h = img_h * bbox[3]

        real_bboxes.append([w, h, center_x, center_y])
    return real_bboxes, lbles


def yolo2xyxy(output, split=7, img_w=448, img_h=448):
    real_bboxes, lbles = yolo2whctr(output, split=split, img_w=img_w, img_h=img_h)
    bboxes = []
    for bbox in real_bboxes:
        w, h, center_x, center_y = bbox
        x1, y1, x2, y2 = center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2
        bboxes.append([x1, x2, y1, y2])
    return bboxes, lbles


def iou(logit, label):
    A = Anchor(logit[0], logit[1], logit[2], logit[3])
    B = Anchor(label[0], label[1], label[2], label[3])

    inner_w = min(A.x2, B.x2) - max(A.x1, B.x1)
    inner_h = min(A.x2, B.x2) - max(A.x1, B.x1)
    inner = 0 if inner_h * inner_w <= 0 else inner_h * inner_w

    outer = (A.x2 - A.x1) * (A.y2 - A.y1) + (B.x2 - B.x1) * (B.y2 - B.y1) - inner

    return inner / outer


def calculate_accuracy(logits, targets):
    return None


def draw_plot_images(train_acc_list: list,
                     value_acc_list: list,
                     train_loss_list: list,
                     value_loss_list: list,
                     save_dir: str):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    plt.plot(train_acc_list, 's--', color='r', label='train_acc')
    plt.plot(value_acc_list, 'o--', color='g', label='value_acc')
    plt.xlabel('length')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'acc.jpg'))
    plt.show()

    plt.plot(train_loss_list, 's--', color='r', label='train_loss')
    plt.plot(value_loss_list, 'o--', color='g', label='value_loss')
    plt.xlabel('length')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.jpg'))
    plt.show()


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predict label')


def create_confuse_matrix(logits, labels):
    if type(logits) is torch.Tensor:
        logits = np.asanyarray(logits)
    if type(labels) is torch.Tensor:
        labels = np.asanyarray(labels)
    if np.unique(logits) is [0, 1]:
        logits = np.argmax(logits, -1).view(-1)
    if np.unique(labels) is [0, 1]:
        labels = np.argmax(labels, -1).view(-1)

    return confusion_matrix(labels, logits)

# debug
# draw_plot_images([1, 2, 3, 4, 5],
#                  [2, 3, 4, 5, 6],
#                  [5, 4, 3, 2, 1],
#                  [6, 5, 4, 3, 2],
#                  './ckpt_dir'
#                  )

# print(iou([1, 1, 500, 500], [1, 1, 500, 500]))

# a = torch.randint(1, 10, [300])
# test_y =torch.randint(1, 10, [1000])
# pred_y = torch.randint(1, 10, [1000])
#
# cm = confusion_matrix(torch.cat([a, test_y], -1), torch.cat([a, pred_y], -1))
# print(cm)
#
# plot_confusion_matrix(cm, labels_name=[1, 2, 3, 4, 5, 6, 7], title='tst')
# plt.show()

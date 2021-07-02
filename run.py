#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 下午4:54
# @Author  : wenjie.Liu
# @Site    :
# @File    : run.py
# @Software: PyCharm

import torch
from myutil import *
import os
from generate_data import train_loader, value_loader
from net import *


def run(args):
    model = Yolo_v1()
    if not os.path.isdir('/ckpt'):
        os.mkdir('ckpt')
    else:
        ckpt = torch.load('latest.ckpt')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['ckpt'])
    if args.is_train:
        train(train_loader, args, model.train())
    else:
        eval(value_loader, args, model.eval())


def train(dataloader, args, model):
    epochs = args.epoches
    batch_size = args.batch_size
    train


def eval(dataloder, args, model):
    return None

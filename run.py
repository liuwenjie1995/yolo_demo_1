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
from torch.utils import data
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


def value(dataloader, args, model):
    total_loss = 0
    total_acc = 0
    length = len(dataloader.dataset)
    for step, [imgs, targets] in enumerate(dataloader):
        logits = model(imgs)
        total_loss += loss_fn(logits, targets).item()
        total_acc += accury_fn(logits, targets).item()
    return total_loss / length, total_acc / length


def test(dataloder, args, model):
    return None


def train(args, train_dataloader: data.DataLoader, eval_dataloader: data.DataLoader, model: nn.Module):
    epochs = args.epoches
    train_acc_list = []
    value_acc_list = []
    train_loss_list = []
    value_loss_list = []

    # batch_size = args.batch_size
    if args.optim is 'adam':
        optimizer = torch.optim.adam.Adam
    elif args.optim is 'sgd':
        optimizer = torch.optim.sgd.SGD
    else:
        optimizer = torch.optim.adamw.AdamW

    train_length = len(train_dataloader.dataset)
    for epoch in range(epochs):
        # train status
        train_loss = 0
        train_acc = 0
        for step, [imgs, targets] in enumerate(train_dataloader):
            logits = model(imgs)
            loss = loss_fn(logits, targets)
            acc = accury_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc.item()
            if step % 100 is 0 and step is not 0:
                print('[epoch:{%5d}, step:{%5d}]--------loss is {%5.5f}, accury is {%2.5f} -------------', epoch, step,
                      loss.item, acc)

        train_loss = train_loss / train_length
        train_acc = train_acc / train_length

        # eval status
        value_loss, value_acc = value(eval_dataloader, args, model.eval())

        ckpt = {'epoch': epoch,
                'train_loss': train_loss,
                'value_loss': value_loss,
                'train_acc': train_acc,
                'value_acc': value_acc,
                'ckpt': model.state_dict(),
                }

        if not os.path.isdir('ckpt_dir'):
            os.mkdir('ckpt_dir')
        torch.save(ckpt, 'ckpt_dir/latest.ckpt')
        if value_acc >= max(value_acc_list):
            torch.save(ckpt, 'ckpt_dir/best.ckpt')

        draw_plots(train_acc_list, value_acc_list, train_loss_list, value_loss_list, save_dir)

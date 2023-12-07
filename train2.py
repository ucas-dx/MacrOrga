# -*- coding: utf-8 -*-
# @Time : 2023/12/7 16:18
# @Author : Dengxun
# @Email : 38694034@qq.com
# @File : train2.py
# @Project : orgseg
from torch import optim
from torch.optim.lr_scheduler import StepLR
import tqdm

from utils import *
from models.ResnetUnet import unetmodel
from data import Val_loader,train_loader
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import tqdm

def train_loop(**kwargs):
    device = kwargs.get('device', 'cuda')
    train_loader = kwargs.get('train_loader')
    val_loader = kwargs.get('val_loader')
    num_epochs = kwargs.get('num_epochs', 150)
    learning_rate = kwargs.get('learning_rate', 0.01)
    step = kwargs.get('step', 50)
    momentum = kwargs.get('momentum', 0.9)
    nesterov = kwargs.get('nesterov', True)
    weight_decay = kwargs.get('weight_decay', 0.00001)

    model = unetmodel.to(device)
    print(f"device:{device}-epochs:{num_epochs}-learning_rate:{learning_rate}")

    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    criterion3 = FocalLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        train_num_acc = 0

        for batch in tqdm.tqdm(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            optimizer.zero_grad()

            loss1 = criterion1(outputs, masks)
            loss2 = criterion2(outputs, masks)
            loss3 = criterion3(outputs, masks)
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()
            train_iou += iou(torch.sigmoid(outputs), masks).item()

        scheduler.step()
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        train_num_acc /= len(train_loader)

        def val(data_loader):
            model.eval()
            val_loss = 0
            val_dice = 0
            val_iou = 0
            with torch.no_grad():
                for batch in tqdm.tqdm(data_loader):
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    outputs = model(images)
                    optimizer.zero_grad()

                    loss1 = criterion1(outputs, masks)
                    loss2 = criterion2(outputs, masks)
                    loss3 = criterion3(outputs, masks)
                    loss = loss1 + loss2 + loss3

                    val_loss += loss.item()
                    val_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()
                    val_iou += iou(torch.sigmoid(outputs), masks).item()

            val_loss /= len(data_loader)
            val_dice /= len(data_loader)
            val_iou /= len(data_loader)
            return val_loss, val_dice, val_iou

        val_loss, val_dice, val_iou = val(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')

        torch.save(model.state_dict(), f'Epoch_{epoch + 1}_modeldict.pth')

# Example usage:
kwargs = {
    'device': 'cuda',
    'train_loader': train_loader,
    'val_loader': Val_loader,
    'num_epochs': 150,
    'learning_rate': 0.01,
    'step': 50,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 0.00001
}

train_loop(**kwargs)


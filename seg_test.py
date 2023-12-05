#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# @Time : 2023/12/4 22:22
# @Author : Dengxun
# @Email : 38694034@qq.com
# @Software: PyCharm 
# -------------------------------------------------------------------------------
import argparse
import time
import torch
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from models.MacrOrga import MacrOrganet
from models.ResnetUnet import unetmodel
from utils import *
from data import test_loader
def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Model Inference')
    parser.add_argument('--model_name', type=str, default='MacrOrga', help='model_name')
    parser.add_argument('--model_path', type=str, default='MacrOrga.pth', help='Path to the trained model file')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference (cuda or cpu)')
    parser.add_argument('--imgshow', default=True, help='Show images during inference')
    return parser.parse_args()


def modelseg(model_name,model_path, Model,device='cuda', test_loader=test_loader, imgshow=True):
    model =Model
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    total_time = 0
    for epoch in range(1):
        model.eval()
        test_dice = 0
        test_iou = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                total_time += end_time - start_time
                test_dice += dice_coefficient(torch.sigmoid(outputs), masks).item()
                test_iou += iou(torch.sigmoid(outputs), masks).item()

                if imgshow:
                    # 获取原始图像和模型输出的图像
                    original_images = images.cpu().numpy()
                    segmentation_results = torch.sigmoid(outputs).cpu().numpy()
                    # 显示图像并将原始图像与分割结果拼接在一起
                    for i in range(images.shape[0]):
                        original_image = np.transpose(original_images[i], (1, 2, 0))
                        segmentation_result = segmentation_results[i, 0, :, :]
                        # 将分割结果转换为二进制掩码
                        segmentation_mask = (segmentation_result > 0.5).astype(np.uint8)
                        # 将原图和分割结果水平拼接
                        concatenated_image = np.hstack((original_image, segmentation_mask[:, :, np.newaxis]))
                        # 显示拼接后的图像
                        plt.imshow(concatenated_image)
                        plt.axis('off')
                        plt.show()

        test_dice /= len(test_loader)
        test_iou /= len(test_loader)
        print(f'Test by using {model_name}-, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}')

    avg_time_per_image = total_time / len(test_loader.dataset)
    print(f'Average inference time per image by using {model_name}: {avg_time_per_image:.4f} seconds')


if __name__ == "__main__":
    args = parse_args()
    model_dict={"MacrOrga":{'model':MacrOrganet(),'model_dict':'MacrOrga.pth'},"ResnetUnet":{'model':unetmodel,'model_dict':'resnet_unet.pth'}}
    choose=list(model_dict.keys())[0]#0 or 1 选择MacrOrga还是ResnetUnet
    modelseg(model_name=choose,model_path=model_dict.get(choose)['model_dict'],Model=model_dict.get(choose)['model'], device=args.device, test_loader=test_loader, imgshow=args.imgshow)

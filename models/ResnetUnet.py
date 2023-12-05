# -*- coding: utf-8 -*-
# @Time : 2023/12/4 23:45
# @Author : Dengxun
# @Email : 38694034@qq.com
# @File : ResnetUnet.py
# @Project : orgunet
import segmentation_models_pytorch as smp
# 定义 UNet 模型
unetmodel = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,  # 输出通道数，根据你的任务设置
)
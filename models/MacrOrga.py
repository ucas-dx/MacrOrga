# -*- coding: utf-8 -*-
# @Time : 2023/12/4 22:22
# @Author : Dengxun
# @Email : 38694034@qq.com
# @File : MacrOrga.py
# @Project : orgunet
import timm
from timm.models.resnet import  resnet50
import torch.nn as nn
import os
import torch
import numpy as np
import random
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
np.random.seed(seed)
torch.manual_seed(seed)


class decodeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decodeConv, self).__init__()
        self.quadruple_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d( out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, 2 * out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(2* out_channels),
            nn.GELU(),
            nn.Conv2d(2*out_channels, 1 * out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(1 * out_channels),
        )

    def forward(self, x):
        x = x
        x1 = self.quadruple_conv1(x)
        return x1


class Rest50(nn.Module):
    def __init__(self):
        super().__init__()
        model=resnet50(pretrained=False)
        model=nn.Sequential(*list(model.children()))
        self.model_down=model[0:3]
        self.stage1= model[4][0]
        self.stage2 = model[5][0]
        self.stage3 = model[6][0]
        self.stage4 = model[7][0]
    def forward(self,x):
        x_256=self.model_down(x)
        #print(x_256.shape)
        x_256=self.stage1(x_256)
        #print(x_256.shape)
        x_128=self.stage2(x_256)
        #print(x_128.shape)
        x_64 = self.stage3(x_128)
        #print(x_64.shape)
        x_32=self.stage4(x_64)
        #print(x_32.shape)
        return x_256, x_128, x_64,x_32
class max_vit_encode(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.models.create_model('maxvit_large_tf_512', pretrained=False)
        self.down=model.stem
        #self.stem =swinBlock(128,in_channel_last=False,block_num=2) #model_train.stem
        self.stages1 = (model.stages[0].blocks)[0:1]
        self.stages2 = (model.stages[1].blocks)[0:1]
        self.stages3 = (model.stages[2].blocks)[0:1]
        self.stages4 = (model.stages[3].blocks)[0:1]
    def forward(self,x):
        x_256=self.down(x)
       #x_256=self.stem(x_256)#.transpose(-1,-3)
        x_128=self.stages1(x_256)
        x_64=self.stages2(x_128)
        x_32 = self.stages3(x_64)
        x_16=self.stages4(x_32)
        return x_128, x_64, x_32,x_16

class up_feature(nn.Module):
    def __init__(self,inc,outc,scale,channel_last=True,out_channel_last=True):
        super(up_feature,self).__init__()
        self.channel_last=channel_last
        self.outchannel_last=out_channel_last
        self.quadruple_conv1 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=1))
        self.layer_norm = nn.LayerNorm(outc)
        self.act = nn.GELU()
        #self.upsam = nn.ConvTranspose2d(inc,outc,kernel_size=scale,stride=scale)
        self.upsam =nn.Sequential(nn.Upsample(scale_factor=scale,mode="bilinear"),nn.Conv2d(inc,outc,kernel_size=1,stride=1),nn.GELU()) #nn.ConvTranspose2d(inc, outc, kernel_size=scale, stride=scale)
    def forward(self,x):
        if self.channel_last==True:
            x = x.permute(0, 3, 1, 2)
        else:pass
        x = self.upsam(x)
        # x=self.quadruple_conv1(x).permute(0,2,3,1)
        # x=self.layer_norm(x).permute(0,3,1,2)
        # x=self.act(x)
        if self.outchannel_last==True:
            x=x.permute(0,2,3,1)
        else:pass
        return x

class MSCF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding="same"),
                                 nn.BatchNorm2d(16))
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding="same"),
                                 nn.BatchNorm2d(16))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding="same"))
        self.concatenate_conv=nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding="same"),
                                            nn.BatchNorm2d(3),nn.GELU())
    def forward(self,x):
        xcat=torch.cat([x,x,x],dim=1)
        x1=self.conv1(x)
        x3=self.conv3(x)
        x5=self.conv5(x)
        concat=torch.cat([x3,x5],dim=1)
        out=self.concatenate_conv(concat)+x1
        return xcat+out*xcat

class channelcat(nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channels=inc,out_channels=outc,stride=1,kernel_size=1,padding="same"),nn.BatchNorm2d(outc))
    def forward(self,x):
        x=self.conv(x)
        return x
class MacrOrganet(nn.Module):
    def __init__(self):
        super(MacrOrganet, self).__init__()
        self.rgb=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=3,stride=1,kernel_size=3,padding="same"),nn.BatchNorm2d(3))
        self.max_vit_encode=max_vit_encode()
        self.resnet50=Rest50()
        self.up_feature_16_32 = up_feature(1024, 512, 2, channel_last=False, out_channel_last=False)
        self.up_feature_32_64 = up_feature(512, 256, 2,channel_last=False,out_channel_last=False)
        self.cat32=channelcat(512+2048,512)
        self.decode32=decodeConv(512,512)#512,32,32->512,32,32
        #self.up_feature_32_512 = up_feature(512, 64, 16,channel_last=False,out_channel_last=False)  # 512,32,32->128,512,512
        self.up_feature_64_128 = up_feature(256, 128, 2,channel_last=False,out_channel_last=False)
        self.cat64 = channelcat(256+1024, 256)
        self.decode64 = decodeConv(256, 256)
        #self.up_feature_64_512 = up_feature(256, 64,8,channel_last=False,out_channel_last=False)  # 256,64,64->256,64,64
        self.cat128 = channelcat(128+512, 128)
        self.decode128 = decodeConv(128, 128)
        self.decode256 = decodeConv(128+256, 128)
        self.up_feature_128_256 = up_feature(128, 128,2,channel_last=False,out_channel_last=False)
        self.up_feature_256_512 = up_feature(128, 64, 2, channel_last=False, out_channel_last=False)
        #self.up_feature_256_512_1 = nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear"),nn.Conv2d(96,1,kernel_size=1,stride=1),nn.GELU())
        self.out=nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),
                               nn.GELU(),nn.Conv2d(64,1,1))

    def forward(self,x):
        #x=self.rgb(x)
        x = torch.cat([x, x, x], dim=1)
        x_swin=self.max_vit_encode(x)
        #x_256=self.encode256(x)
        x_128,x_64,x_32,x_16 =x_swin
        x_256res, x_128res, x_64res, x_32res=self.resnet50(x)
        x_cat32 = torch.cat([x_32res, x_32], dim=1)
        x_cat32=self.cat32(x_cat32)
        x_16up32=self.up_feature_16_32(x_16)+x_cat32
        x32 = self.decode32(x_16up32)
        x_cat64 = torch.cat([x_64res, x_64], dim=1)
        x_cat64=self.cat64(x_cat64)
        x_32up64=self.up_feature_32_64(x32)+x_cat64
        x64 = self.decode64(x_32up64)
        x_cat128 = torch.cat([x_128res, x_128], dim=1)
        x_cat128=self.cat128(x_cat128)
        x_64up128=self.up_feature_64_128(x64)+x_cat128
        x128=self.decode128(x_64up128)
        x_128up256=self.up_feature_128_256(x128)
        x_128up256=torch.cat([x_256res,x_128up256],dim=1)
        x256=self.decode256(x_128up256)
        # x256=torch.cat([x256,x_256],dim=1)
        x_512=self.up_feature_256_512(x256)
        # x_256_512=self.up_feature_256_512(x256)
        out = self.out(x_512)
        return out
# if __name__=="__main__":
#     model=MaxVit_ConvParallel()
#     model.load_state_dict(torch.load(r'/MacrOrga.pth', map_location='cuda'))
# Editor       : pycharm
# File name    : models/yolov3.py
# Author       : huangxinyu
# Created date : 2020-11-06
# Description  : yolov3的模型定义

import torch
import torch.nn as nn
from config.yolov3 import cfg
from utils.decode import transform,build_decode


class convolution(nn.Module):
    '''
    in_channel:输入通道数
    out_channel:输出通道数
    kernel_size:卷积核大小
    stride:卷积布长
    padding:补空
    if_bn:是否使用batchnorm2d
    if_activity:是否使用激活函数
    '''
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding,if_bn,if_activity,if_pooling=False):
        super(convolution, self).__init__()
        self.if_bn=if_bn
        self.if_activity = if_activity
        self.if_pooling = if_pooling
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, stride=stride,padding=padding,bias= not if_bn)
        if if_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channel,momentum=0.9,eps=1e-5)
        if if_activity:
            self.activity = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        if if_pooling:
            self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x = self.pooling(x) if self.if_pooling else x
        if self.if_bn:
            x = self.conv(x)
            x = self.bn(x)
        else:
            x = self.conv(x)
        return self.activity(x) if self.if_activity else x


class residual_block(nn.Module):
    def __init__(self,channel):
        super(residual_block,self).__init__()
        self.conv_1x1 = convolution(in_channel=channel,out_channel=channel//2,kernel_size=1,stride=1,padding=0,if_bn=True,if_activity=True)
        self.conv_3x3 = convolution(in_channel=channel//2,out_channel=channel,kernel_size=3,stride=1,padding=1,if_bn=True,if_activity=True)
    def forward(self,x):
        res = self.conv_1x1(x)
        res = self.conv_3x3(res)
        return res+x


class darknet53(nn.Module):
    def __init__(self):
        super(darknet53,self).__init__()
        self.conv_0 = convolution(in_channel=3,out_channel=32,kernel_size=3,stride=1,padding=1,if_bn=True,if_activity=True)
        self.conv_1 = convolution(in_channel=32,out_channel=64,kernel_size=3,stride=2,padding=1,if_bn=True,if_activity=True)
        self.conv_2 = convolution(in_channel=64,out_channel=128,kernel_size=3,stride=2,padding=1,if_bn=True,if_activity=True)
        self.conv_3 = convolution(in_channel=128,out_channel=256,kernel_size=3,stride=2,padding=1,if_bn=True,if_activity=True)
        self.conv_4 = convolution(in_channel=256,out_channel=512,kernel_size=3,stride=2,padding=1,if_bn=True,if_activity=True)
        self.conv_5 = convolution(in_channel=512,out_channel=1024,kernel_size=3,stride=2,padding=1,if_bn=True,if_activity=True)

        self.residual_1_1 = residual_block(64)

        self.residual_2_1 = residual_block(128)
        self.residual_2_2 = residual_block(128)

        self.residual_3_1 = residual_block(256)
        self.residual_3_2 = residual_block(256)
        self.residual_3_3 = residual_block(256)
        self.residual_3_4 = residual_block(256)
        self.residual_3_5 = residual_block(256)
        self.residual_3_6 = residual_block(256)
        self.residual_3_7 = residual_block(256)
        self.residual_3_8 = residual_block(256)

        self.residual_4_1 = residual_block(512)
        self.residual_4_2 = residual_block(512)
        self.residual_4_3 = residual_block(512)
        self.residual_4_4 = residual_block(512)
        self.residual_4_5 = residual_block(512)
        self.residual_4_6 = residual_block(512)
        self.residual_4_7 = residual_block(512)
        self.residual_4_8 = residual_block(512)

        self.residual_5_1 = residual_block(1024)
        self.residual_5_2 = residual_block(1024)
        self.residual_5_3 = residual_block(1024)
        self.residual_5_4 = residual_block(1024)

    def forward(self,img):    #img(n,3,416,416)
        x = self.conv_0(img)

        x = self.conv_1(x)    #(n,64,208,208)
        x = self.residual_1_1(x)

        x = self.conv_2(x)    #(n,128,104,104)
        x = self.residual_2_1(x)
        x = self.residual_2_2(x)

        x = self.conv_3(x)    #(n,256,52,52)
        x = self.residual_3_1(x)
        x = self.residual_3_2(x)
        x = self.residual_3_3(x)
        x = self.residual_3_4(x)
        x = self.residual_3_5(x)
        x = self.residual_3_6(x)
        x = self.residual_3_7(x)
        x = self.residual_3_8(x)
        out1 = x

        x = self.conv_4(x)
        x = self.residual_4_1(x)
        x = self.residual_4_2(x)
        x = self.residual_4_3(x)
        x = self.residual_4_4(x)
        x = self.residual_4_5(x)
        x = self.residual_4_6(x)
        x = self.residual_4_7(x)
        x = self.residual_4_8(x)
        out2 = x

        x = self.conv_5(x)
        x = self.residual_5_1(x)
        x = self.residual_5_2(x)
        x = self.residual_5_3(x)
        x = self.residual_5_4(x)

        return out1,out2,x


class yolov3(nn.Module):
    def __init__(self):
        super(yolov3,self).__init__()
        self.darknet53 = darknet53()
        self.bobj_stage = nn.Sequential(
            convolution(1024 ,512, 1, 1, 0, True, True),
            convolution(512, 1024, 3, 1, 1, True, True),
            convolution(1024, 512, 1, 1, 0, True, True),
            convolution(512, 1024, 3, 1, 1, True, True),
            convolution(1024, 512, 1, 1, 0, True, True)
        )
        self.bobj_out_stage = nn.Sequential(
            convolution(512 ,1024, 3, 1, 1, True, True),
            convolution(1024, 3*(5 + cfg.num_classes), 1, 1, 0, False, False)
        )
        self.mobj_stage = nn.Sequential(
            convolution(768, 256, 1, 1, 0, True, True),
            convolution(256, 512, 3, 1, 1, True, True),
            convolution(512, 256, 1, 1, 0, True, True),
            convolution(256, 512, 3, 1, 1, True, True),
            convolution(512, 256, 1, 1, 0, True, True)
        )
        self.mobj_stage_conv = convolution(512, 256, 1, 1, 0, True, True)
        self.mobj_out_stage = nn.Sequential(
            convolution(256, 512, 3, 1, 1, True, True),
            convolution(512, 3 * (5 + cfg.num_classes), 1, 1, 0, False, False)
        )
        self.sobj_stage = nn.Sequential(
            convolution(384, 128, 1, 1, 0, True, True),
            convolution(128, 256, 3, 1, 1, True, True),
            convolution(256, 128, 1, 1, 0, True, True),
            convolution(128, 256, 3, 1, 1, True, True),
            convolution(256, 128, 1, 1, 0, True, True)
        )
        self.sobj_stage_conv = convolution(256, 128, 1, 1, 0, True, True)
        self.sobj_out_stage = nn.Sequential(
            convolution(128, 256, 3, 1, 1, True, True),
            convolution(256, 3 * (5 + cfg.num_classes), 1, 1, 0, False, False)
        )
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))

    def forward(self, img):
        route1, route2, x = self.darknet53(img)

        # big object
        x = self.bobj_stage(x)
        bobj_output = self.bobj_out_stage(x)

        # middle object
        x = self.mobj_stage_conv(x)
        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, route2), dim=1)
        x = self.mobj_stage(x)
        mobj_output = self.mobj_out_stage(x)

        # small object
        x = self.sobj_stage_conv(x)
        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat((x, route1), dim=1)
        x = self.sobj_stage(x)
        sobj_output = self.sobj_out_stage(x)

        output = [transform(sobj_output), transform(mobj_output), transform(bobj_output)]
        pred = build_decode(output)
        return output, pred

if __name__ == '__main__':
    model=yolov3()
    input = torch.randn((1,3,416,416))
    sobj_output,mobj_output,bobj_output = model(input)
    print(sobj_output.shape)
    print(mobj_output.shape)
    print(bobj_output.shape)




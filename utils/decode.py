# Editor       : pycharm
# File name    : utils/decode.py
# Author       : huangxinyu
# Created date : 2020-11-10
# Description  : 对yolov3的输出进行解码

import torch
from config.yolov3 import cfg

def transform(output):
    num_classes = cfg.num_classes
    batch_size, _, output_size, _ = output.shape
    num_anchors = len(cfg.anchors[0])
    output = output.permute(0, 2, 3, 1).contiguous()  # (batch size,3*(5+num classes),output size,output size)
    output = output.view(batch_size, output_size, output_size, num_anchors, num_classes + 5)

    return output

def build_decode(output):
    anchors = torch.FloatTensor(cfg.anchors)
    strides = torch.FloatTensor(cfg.strides)
    small_pred = decode(output[0], strides[0], anchors[0])
    middle_pred = decode(output[1], strides[1], anchors[1])
    big_pred = decode(output[2], strides[2], anchors[2])

    return small_pred,middle_pred,big_pred

def decode(output,stride,anchors):
    decice = torch.device(cfg.device)
    batch_size,output_size = output.shape[0:2]
    anchors = anchors.to(torch.device(cfg.device))

    output_xy = output[...,0:2]      #中心点x和y
    output_wh = output[...,2:4]      #w和h
    output_conf = output[...,4:5]    #置信度
    output_prob = output[...,5:]     #概率分布

    y_stride = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size).to(torch.float32) #每个网格y的偏移量
    x_offset = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1).to(torch.float32) #每个网格x的偏移量
    xy_offset = torch.stack([x_offset, y_stride], dim=-1)
    xy_offset = xy_offset.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).to(decice)

    output_xy = (torch.sigmoid(output_xy)+xy_offset)*stride    #x和y加上偏移量并乘以stride

    output_wh = (torch.exp(output_wh)*anchors)*stride     #w和h乘以三种不同的anchors并乘以stride
    output_conf = torch.sigmoid(output_conf)
    output_prob = torch.sigmoid(output_prob)

    pred = torch.cat((output_xy,output_wh,output_conf,output_prob),-1)
    return pred
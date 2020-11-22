# Editor       : pycharm
# File name    : utils/loss.py
# Author       : huangxinyu
# Created date : 2020-11-15
# Description  : 计算训练loss

import torch
import torch.nn as nn
from config.yolov3 import cfg


def box_iou(box1, box2, giou=False):
    '''
    :param box1: shape(...,(x,y,w,h))
    :param box2: shape(...,(x,y,w,h))
    :param giou: 是否计算giou
    :return:
        iou or giou
    '''
    box1_area = box1[..., 2:3] * box1[..., 3:4]  # 计算box1的面积
    box2_area = box2[..., 2:3] * box2[..., 3:4]  # 计算box2的面积

    #转为x1,y1,x2,y2
    box1 = torch.cat((box1[..., 0:2] - box1[..., 2:] * 0.5, box1[..., 0:2] + box1[..., 2:] * 0.5), dim=-1)
    box2 = torch.cat((box2[..., 0:2] - box2[..., 2:] * 0.5, box2[..., 0:2] + box2[..., 2:] * 0.5), dim=-1)

    left_up = torch.max(box1[..., :2],box2[..., :2])  # 求两个box的左上角顶点的最右下角点
    right_down = torch.min(box1[..., 2:],box2[..., 2:])  # 求两个box的右下角顶点的最左上角点


    inter = right_down - left_up
    zero = torch.zeros((inter.shape),dtype=torch.float32).to(torch.device(cfg.device))
    inter = torch.max(inter, zero)  # 计算横向和纵向的交集长度,如果没有交集则为0
    inter = inter[...,0:1] * inter[...,1:2]  # 计算交集面积

    union = box1_area + box2_area - inter  # 计算并集面积
    iou = 1.0 * inter / union    # iou = 交集/并集

    if giou:
        left_up = torch.min(box1[..., :2], box2[..., :2])  # 求两个box的左上角顶点的最左上角点
        right_down = torch.max(box1[..., 2:], box2[..., 2:])  # 求两个box的右下角顶点的最右下角点
        area_c = right_down - left_up
        area_c = area_c[...,0:1] * area_c[...,1:2]   #计算两个box的最小外接矩形的面积
        giou = iou - (area_c - union) / area_c
        return giou
    else:
        return iou

def Focal_loss(input,target,gamma,alpha):
    BCE = nn.BCEWithLogitsLoss(reduction='none')
    loss = BCE(input,target)
    loss *= alpha*torch.pow(torch.abs(target-torch.sigmoid(input)),gamma)
    return loss

def BCE_loss(input,target):
    BCE = nn.BCEWithLogitsLoss(reduction='none')   #计算交叉熵之前会对input做sigmoid,所以不用提前经过sigmoid
    loss = BCE(input,target)
    return loss

def loss_layer(output,pred,label_mask,label_xywh,stride):
    '''
    :param output: yolo output(n,grid size,grid size,num anchors,5+num classes)
    :param pred: yolo output before decode(n,grid size,grid size,num anchors,5+num classes)
    :param label_mask: shape same as output and pred
    :param label_xywh: (max num of boxes every scale,(x,y,w,h))
    :param stride: input size//putput size
    :return:
        loss,giou_loss,conf_loss,cls_loss
    '''

    batch_size,output_size = output.shape[0:2]    #batch_size和yolo输出的大小
    input_size = output_size*stride   #输入的大小

    output_conf = output[...,4:5]
    output_cls = output[...,5:]

    pred_xywh = pred[...,0:4]         #预测的每个ceil里的目标的xywh

    mask_xywh = label_mask[...,0:4]
    mask_conf = label_mask[...,4:5]   #label每个ceil是否有目标的置信度,有标签的为1,无标签的为0
    mask_cls = label_mask[...,5:]

    # giou loss
    giou = box_iou(pred_xywh,mask_xywh,giou=True)
    bbox_loss = 2.0 - 1.0 * mask_xywh[:, :, :, :, 2:3] * mask_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = mask_conf * bbox_loss * (1 - giou)

    # conf loss
    iou = box_iou(pred_xywh.unsqueeze(4),label_xywh.unsqueeze(1).unsqueeze(1).unsqueeze(1),
                  giou=False).squeeze(-1)    #(n,size,size,num anchors,150)
    iou_max = iou.max(-1, keepdim=True)[0]   #(n,size,size,num anchors,1)
    label_noobj_mask = (1.0 - mask_conf) * (iou_max < cfg.iou_thresh)
    conf_loss = (mask_conf * Focal_loss(input=output_conf,target=mask_conf,gamma=2,alpha=1) +
                 label_noobj_mask * Focal_loss(input=output_conf,target=mask_conf,gamma=2,alpha=1))

    # cls loss
    cls_loss = mask_conf * BCE_loss(output_cls,mask_cls)

    giou_loss = torch.sum(giou_loss)/batch_size
    conf_loss = torch.sum(conf_loss)/batch_size
    cls_loss = torch.sum(cls_loss)/batch_size
    loss = giou_loss+conf_loss+cls_loss

    return loss,giou_loss,conf_loss,cls_loss


def build_loss(output,pred,small_mask,middle_mask,big_mask,small_xywh,middle_xywg,big_xywh):
    #计算每种scale的loss
    loss_small = loss_layer(output[0],pred[0],small_mask,small_xywh,cfg.strides[0])
    loss_middle = loss_layer(output[1],pred[1],middle_mask,middle_xywg,cfg.strides[1])
    loss_big = loss_layer(output[2],pred[2],big_mask,big_xywh,cfg.strides[2])

    giou_loss = loss_small[1] + loss_middle[1] + loss_big[1]
    conf_loss = loss_small[2] + loss_middle[2] + loss_big[2]
    cls_loss = loss_small[3] + loss_middle[3] + loss_big[3]
    loss = loss_small[0] + loss_middle[0] + loss_big[0]

    return loss,giou_loss,conf_loss,cls_loss


if __name__ == '__main__':

    output = torch.randn((1,13,13,3,6)).to(torch.device(cfg.device))
    pred = torch.randn((1,13,13,3,6)).to(torch.device(cfg.device))
    mask = torch.randn((1,13,13,3,6)).to(torch.device(cfg.device))
    xywh = torch.randn((1,150,4)).to(torch.device(cfg.device))
    stride = 32
    loss,giou_loss,conf_loss,cls_loss = loss_layer(output,pred,mask,xywh,stride)
    print(loss)








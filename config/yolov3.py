# Editor       : pycharm
# File name    : config/yolov3.py
# Author       : huangxinyu
# Created date : 2020-11-05
# Description  : 超参数
from easydict import EasyDict

cfg = EasyDict()

# custom
cfg.annotations_path       = "./data/annotations/annotations.txt"   #标签的path
cfg.class_path             = "./data/data.names"                    #类别名文件
cfg.image_path             = "./data/images/"                       #存放图像的path
cfg.mean_and_val           = "./data/mean_and_val.txt"              #数据集均值和方差
cfg.tensorboard_path       = "./log/"                               #存放tensorboard的log输出
cfg.checkpoint_save_path   = "./checkpoint/"                        #存放训练参数
cfg.num_classes            = 1                                      #有多少类
cfg.strides                = [8,16,32]                              #输入与三个分支的大小比例
cfg.device                 = "cuda"                                 #cpu
cfg.anchors                = [[[1.25,1.625],[2.0,3.75],[4.125,2.875]],
                              [[1.875,3.8125],[3.875,2.8125],[3.6875,7.4375]],
                             [[3.625,2.8125],[4.875,6.1875],[11.65625,10.1875]]]


# train
cfg.batch_size             = 2        #每次训练的batch size
cfg.input_sizes            = [320,352,384,416,448,480,512,544,576,608]   #随机选择的输入图像大小
cfg.max_boxes_per_scale    = 150      #label每个scale最多有多少个box
cfg.if_pad                 = True     #对输入resize是否进行补空
cfg.random_horizontal_flip = True     #随机水平翻转
cfg.random_crop            = True     #随机裁剪
cfg.max_epoch              = 300      #最多学习的epoch数
cfg.lr_start               = 1e-4     #初始學習率
cfg.lr_end                 = 1e-6     #結束學習率
cfg.warmup                 = 200      #前多少iter採取warmup測略
cfg.momentum               = 0.9      #动量参数
cfg.weight_decay           = 0.0005   #权重衰减正则项防止过拟合
cfg.iou_thresh             = 0.225    #计算loss时的iou thresh
cfg.focal_gamma            = 2        #计算conf loss的focal loss的gamma参数
cfg.focal_alpha            = 0.5      #计算conf loss的focal loss的alpha参数


# test
cfg.test_input_size        = 416      #输入大小
cfg.conf_thresh            = 0.3
cfg.cls_thresh             = 0.5
cfg.nms_thresh             = 0.2


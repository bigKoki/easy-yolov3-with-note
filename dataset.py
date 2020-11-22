# Editor       : pycharm
# File name    : dataset.py
# Author       : huangxinyu
# Created date : 2020-11-05
# Description  : 加载训练和测试数据

import numpy as np
from config.yolov3 import cfg
import torch
import random
import cv2


class dataloader(object):
    def __init__(self):
        self.image_path = cfg.image_path                 #图像保存路径
        self.annotations_path = cfg.annotations_path     #标签保存路径
        self.class_path = cfg.class_path                 #类别名保存路径
        self.class_names = self.get_class_names()        #类别名
        self.num_classes = len(self.class_names)         #类别数
        self.bacth_size = cfg.batch_size                 #batch size
        self.anchors = np.array(cfg.anchors)             #三种不同尺度的三种anchors,一共九个
        self.annotations = self.get_annotations()        #图像名和属于此图的bboxes
        self.num_annotations = len(self.annotations)     #样本数量
        self.num_batches = np.ceil(len(self.annotations)/self.bacth_size)     #一个epoch有多少个batch
        self.input_sizes = cfg.input_sizes               #一个list,从中随机选取输入图像大小
        self.output_size = [52,26,13]                    #yolo输出大小,根据input size来计算
        self.strides = cfg.strides
        self.max_boxes_per_scale = cfg.max_boxes_per_scale

        self.iter = 0                                    #当前迭代次数

    def __iter__(self):
        return self

    def __next__(self):
        input_size = random.choice(self.input_sizes)   #每次随机选取输入图像的大小
        self.output_size = [input_size//stride for stride in self.strides]    #yolo输出大小
        batch_images = np.zeros((self.bacth_size,input_size,input_size,3)).astype(np.float32)
        batch_mask_bboxes = [np.zeros((self.bacth_size,self.output_size[i],self.output_size[i],
                                      len(self.anchors[0]), 5 + self.num_classes)).astype(np.float32) for i in range(3)]
        batch_list_bboxes = [np.zeros((self.bacth_size,self.max_boxes_per_scale,4)).astype(np.float32) for _ in range(3)]
        annotation_count = 0    #这个batch已经处理了多少个annotation
        if self.iter<self.num_batches:    #迭代次数小于一个epoch的batch数量
            while annotation_count < self.bacth_size:    #已处理的annotation数量小于batch size
                index = self.iter*self.bacth_size + annotation_count   #计算annotation的index
                index = index if(index < self.num_annotations) else (index-self.num_annotations)  #如果index大于样本量,则从样本第一个开始继续取

                image_and_labels = self.annotations[index]    #取image name和labels
                image = self.get_image_array(image_and_labels[0])    #image -> np.array
                bboxes = self.get_bbox_array(image_and_labels[1:])   #str -> np.array

                image, bboxes = self.data_augmentation(image,bboxes)   #数据增强
                image, bboxes = self.resize_image(image,bboxes,input_size)        #resize到随机随机选取的图像大小
                image = self.normalization(image)               #归一化以加快收敛速度
                batch_images[annotation_count] = image          #预处理后的image放入batch
                label = self.extract_label(bboxes)              #对bbox进行处理
                batch_mask_bboxes[0][annotation_count] = label[0][0]
                batch_mask_bboxes[1][annotation_count] = label[0][1]
                batch_mask_bboxes[2][annotation_count] = label[0][2]
                batch_list_bboxes[0][annotation_count] = label[1][0]
                batch_list_bboxes[1][annotation_count] = label[1][1]
                batch_list_bboxes[2][annotation_count] = label[1][2]
                #self.show_image_and_bboxes(np.copy(image), np.copy(bboxes))    #可视化查看数据增强的正确性
                annotation_count += 1      #一个batch里已处理的数目加一
            self.iter += 1
            batch_images = batch_images.transpose([0,3,1,2])    #转置成(n,c,h,w)
            batch_images = torch.from_numpy(batch_images)       #转为tensor
            batch_mask_small_bboxes = torch.from_numpy(batch_mask_bboxes[0])
            batch_mask_middle_bboxes = torch.from_numpy(batch_mask_bboxes[1])
            batch_mask_big_bboxes = torch.from_numpy(batch_mask_bboxes[2])
            batch_list_small_bboxes = torch.from_numpy(batch_list_bboxes[0])
            batch_list_middle_bboxes = torch.from_numpy(batch_list_bboxes[1])
            batch_list_big_bboxes = torch.from_numpy(batch_list_bboxes[2])
            return batch_images,batch_mask_small_bboxes,batch_mask_middle_bboxes,batch_mask_big_bboxes,\
                   batch_list_small_bboxes,batch_list_middle_bboxes,batch_list_big_bboxes

        else:
            self.iter = 0     #重置迭代次数
            np.random.shuffle(self.annotations)   #将annotation打乱
            raise StopIteration

    def extract_label(self,bboxes):
        mask_bboxes = [np.zeros((self.output_size[i], self.output_size[i],len(self.anchors[0]), 5 + self.num_classes)) for i in range(3)]
        list_bboxes_xywh = [np.zeros((self.max_boxes_per_scale,4)) for _ in range(3)]
        target_boxes_count = [0,0,0]

        for bbox in bboxes:
            bbox_xyxy = bbox[:4]  # 坐标信息x1,y1,x2,y2
            bbox_class = bbox[4]  # class标签

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class] = 1.0

            # label smoothing:减少真实样本标签的类别在计算损失函数时的权重，最终起到抑制过拟合的效果。
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # 把x1,y1,x2,y2转换为中心点x,y和w,h
            bbox_w = bbox_xyxy[2]-bbox_xyxy[0]
            bbox_h = bbox_xyxy[3]-bbox_xyxy[1]
            bbox_xywh = [bbox_xyxy[0]+bbox_w*0.5,bbox_xyxy[1]+bbox_h*0.5,bbox_w,bbox_h]
            # 把[x,y,w,h]/[8,16,32]
            bbox_xywh_every_scale = np.array([np.divide(bbox_xywh,stride) for stride in self.strides])

            iou = []
            exist_positive = False

            # 对每种大小的anchor做如下处理
            for i in range(3):
                anchors_xywh = np.zeros((len(self.anchors[0]), 4))  # (3,4)
                # 把真实框的x,y放到anchors_xywh里
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_every_scale[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_per_scale = self.box_iou(bbox_xywh_every_scale[i],anchors_xywh)   #计算此scale下的bbox与三种anchor的iou
                iou_thresh = iou_per_scale>0.3     #iou大于thresh的设为true,其他的设为false
                if any(iou_thresh):
                    x = int(bbox_xywh_every_scale[i][0])
                    y = int(bbox_xywh_every_scale[i][1])
                    mask_bboxes[i][y,x,iou_thresh,0:4] = bbox_xywh
                    mask_bboxes[i][y,x,iou_thresh,4] = 1
                    mask_bboxes[i][y,x,iou_thresh,5:] = smooth_onehot

                    list_bboxes_xywh[i][target_boxes_count[i]] = bbox_xywh
                    target_boxes_count[i] += 1

        return mask_bboxes,list_bboxes_xywh

    #对image进行归一化操作
    def normalization(self,image):
        image = image/255.
        return image

    #对图像resize以符合输入要求,可选择pad和no pad方式
    def resize_image(self,image,bboxes,input_size):
        h, w, _ = image.shape    #(h,w,c)
        if not cfg.if_pad:    #直接resize,可能会导致图像变形
            new_image = cv2.resize(image,(input_size,input_size))
            bboxes[:,[0,2]] = bboxes[:,[0,2]]*input_size/w
            bboxes[:,[1,3]] = bboxes[:,[1,3]]*input_size/h
        else:                 #补空保证图像不变形
            scale = input_size/max(w,h)     #得到input size/图像的宽和高较小的那一个scale
            w,h = int(scale*w),int(scale*h)   #将原图像resize到这个大小,不改变原来的形状

            image = cv2.resize(image,(w,h))
            fill_value = 0        #选择边缘补空的像素值
            new_image = np.ones((input_size,input_size,3)) *  fill_value     #新的符合输入大小的图像
            dw,dh = (input_size-w)//2,(input_size-h)//2
            new_image[dh:dh+h,dw:dw+w,:] = image

            bboxes[:, 0] = bboxes[:, 0] * scale + dw
            bboxes[:, 2] = bboxes[:, 2] * scale + dw
            bboxes[:, 1] = bboxes[:, 1] * scale + dh
            bboxes[:, 3] = bboxes[:, 3] * scale + dh

        return new_image,bboxes

    #随机水平翻转
    def random_horizontal_flip(self,image,bboxes):
        flip_image = np.copy(image)
        flip_bboxes = np.copy(bboxes)
        if random.random() < 0.5:
            _, w, _ = image.shape
            flip_image = image[:, ::-1, :]
            flip_bboxes[:,0] = w-bboxes[:,2]
            flip_bboxes[:,2] = w-bboxes[:,0]
        return flip_image,flip_bboxes

    #随机裁剪
    def random_crop(self,image,bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    # 数据增强
    def data_augmentation(self,image,bboxes):
        if cfg.random_horizontal_flip:     #随机水平翻转
            image,bboxes = self.random_horizontal_flip(image,bboxes)
        if cfg.random_crop:                #随机裁剪
            image,bboxes = self.random_crop(image,bboxes)

        return image,bboxes

    # bboxes str -> bboxes array
    def get_bbox_array(self,str_bboxes):
        bboxes_array = []
        for str_box in str_bboxes:
            bboxes_array.append([int(float(x)) for x in str_box.split(",")])
        return np.array(bboxes_array)

    def get_image_array(self,image_name):
        image = np.array(cv2.imread(self.image_path+image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)   #BGR -> RGB
        return image

    # 用于调试,检测数据处理是否正确
    def show_image_and_bboxes(self,image,bboxes):
        image = image.astype(np.uint8)
        for bbox in bboxes:
            cv2.rectangle(image,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),2)
        cv2.imshow("show_image_and_bboxes",image)
        cv2.waitKey(0)

    #读取class的类名
    def get_class_names(self):
        class_names = []
        with open(self.class_path) as class_file:
            classes = class_file.readlines()
            for class_name in classes:
                class_names.append(class_name[:-1])
        return class_names

    #读取标签文件,返回文件名和对应框的list
    def get_annotations(self):
        image_and_coordinate = []
        with open(self.annotations_path) as annotations_file:
            annotations = annotations_file.readlines()
            for annotation in annotations:
                annotation = annotation[:-1].split(" ")
                image_and_coordinate.append(annotation)
        return image_and_coordinate

    # 计算iou
    # input:(x,y,w,h)
    def box_iou(self,box1, box2):
        box1 = np.array(box1)
        box2 = np.array(box2)

        box1_area = box1[..., 2] * box1[..., 3]  # 计算box1的面积
        box2_area = box2[..., 2] * box2[..., 3]  # 计算box2的面积

        left_up = np.maximum(box1[..., :2] - box1[..., 2:] * 0.5,
                             box2[..., :2] - box2[..., 2:] * 0.5)  # 求两个box的左上角顶点的最右下角点
        right_down = np.minimum(box1[..., :2] + box1[..., 2:] * 0.5,
                                box2[..., :2] + box2[..., 2:] * 0.5)  # 求两个box的右下角顶点的最左上角点

        inter = np.maximum(right_down - left_up, 0.0)  # 计算横向和纵向的交集长度,如果没有交集则为0
        inter = inter[..., 0] * inter[..., 1]  # 计算交集面积

        union = box1_area + box2_area - inter  # 计算并集面积

        return inter / union

if __name__ == '__main__':
    d=dataloader()
    for i in (d):
        print(i)
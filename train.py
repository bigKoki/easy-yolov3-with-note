# Editor       : pycharm
# File name    : train.py
# Author       : huangxinyu
# Created date : 2020-11-10
# Description  : 训练

import torch
from torch.autograd import Variable
from models.yolov3 import yolov3
from config.yolov3 import cfg
import math
import time
from tensorboardX import SummaryWriter
from dataset import dataloader
from utils.loss import build_loss
from utils.scheduler import adjust_lr



class trainer(object):
    def __init__(self):
        self.device = torch.device(cfg.device)
        self.max_epoch = cfg.max_epoch
        self.train_dataloader = dataloader()
        self.len_train_dataset = self.train_dataloader.num_annotations
        self.model = yolov3().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=cfg.lr_start,momentum=cfg.momentum,weight_decay=cfg.weight_decay)
        self.scheduler = adjust_lr(self.optimizer,self.max_epoch*self.len_train_dataset,cfg.lr_start,cfg.lr_end,cfg.warmup)
        self.writer = SummaryWriter(cfg.tensorboard_path)
        self.iter = 0

    def put_log(self,epoch_index,mean_loss,time_per_iter):
        print("[epoch:{}|{}] [iter:{}|{}] time:{}s loss:{} giou_loss:{} conf_loss:{} cls_loss:{} lr:{}".format(
            epoch_index + 1, self.max_epoch,
            self.iter + 1, math.ceil(self.len_train_dataset / cfg.batch_size), round(time_per_iter, 2),
            round(mean_loss[0], 4)
            , round(mean_loss[1], 4), round(mean_loss[2], 4), round(mean_loss[3], 4),
            self.optimizer.param_groups[0]['lr']))

        step = epoch_index * math.ceil(self.len_train_dataset / cfg.batch_size) + self.iter
        self.writer.add_scalar("loss", mean_loss[0], global_step=step)
        self.writer.add_scalar("giou loss", mean_loss[1], global_step=step)
        self.writer.add_scalar("conf loss", mean_loss[2], global_step=step)
        self.writer.add_scalar("cls loss", mean_loss[3], global_step=step)
        self.writer.add_scalar("learning rate", self.optimizer.param_groups[0]['lr'], global_step=step)

    def train(self):
        for epoch_index in range(self.max_epoch):
            self.iter = 0
            mean_loss = [0,0,0,0]
            self.model.train()
            for train_data in self.train_dataloader:
                start_time = time.time()
                self.scheduler.step(self.len_train_dataset*epoch_index+self.iter/cfg.batch_size)   #调整学习率

                image,small_mask,middle_mask,big_mask,small_xywh,middle_xywg,big_xywh = train_data

                image =Variable(image).to(self.device)
                small_mask = Variable(small_mask).to(self.device)
                middle_mask = Variable(middle_mask).to(self.device)
                big_mask = Variable(big_mask).to(self.device)
                small_xywh = Variable(small_xywh).to(self.device)
                middle_xywg = Variable(middle_xywg).to(self.device)
                big_xywh = Variable(big_xywh).to(self.device)

                output,pred = self.model(image)

                #计算loss
                loss,loss_giou,loss_conf,loss_cls = build_loss(output,pred,small_mask,middle_mask,big_mask,small_xywh,middle_xywg,big_xywh)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                end_time = time.time()
                time_per_iter = end_time - start_time   #每次迭代所花时间

                loss_items = [loss.item(),loss_giou.item(),loss_conf.item(),loss_cls.item()]
                mean_loss = [(mean_loss[i]*self.iter+loss_items[i])/(self.iter+1) for i in range(4)]
                self.put_log(epoch_index,mean_loss,time_per_iter)
                self.iter += 1

            if (epoch_index+1)%30 == 0:
                torch.save(self.model.state_dict(), cfg.checkpoint_save_path+str(epoch_index+1)+'.pt')



if __name__ == '__main__':
    trainer().train()
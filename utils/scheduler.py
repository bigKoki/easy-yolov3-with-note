# Editor       : pycharm
# File name    : utils/scheduler.py
# Author       : huangxinyu
# Created date : 2020-11-15
# Description  : 动态调整学习率

import numpy as np

class adjust_lr(object):
    def __init__(self,optimizer,iter_max,lr_start,lr_end=0.,warmup=0):
        self.optimizer = optimizer
        self.iter_max = iter_max
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.warmup = warmup

    def step(self,iter):
        if self.warmup and iter < self.warmup:
            lr = self.lr_start / self.warmup * iter
        else:
            T_max = self.lr_start - self.warmup
            iter = iter - self.warmup
            lr = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * (1 + np.cos(iter / T_max * np.pi))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
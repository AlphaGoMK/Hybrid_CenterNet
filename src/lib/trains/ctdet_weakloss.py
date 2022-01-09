import torch
import numpy as np
import torch.nn.functional as F

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class CtCamWeakLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtCamWeakLoss, self).__init__()
        self.opt = opt

        print('Init CtCAM Weak Loss, MultiLabelSoftMarginLoss')

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, cam_aggr_loss = 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            # pool把heatmap变为预测向量，不需要sigmoid
            # multilabel_soft_margin_loss 不需要sigmoid
            hm_loss += F.multilabel_soft_margin_loss(output['hm'], batch['cat_id']) / opt.num_stacks
            cam_aggr_loss += F.multilabel_soft_margin_loss(output['aggregation'], batch['cat_id']) / opt.num_stacks

        if not 'aggregation' in outputs[0]:
            cam_aggr_loss = torch.tensor([0]).float().cuda()

        loss = hm_loss + cam_aggr_loss  # zero out wh/off/cam_off loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'cam_aggr_loss': cam_aggr_loss}
        return loss, loss_stats



class CtdetWeakLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetWeakLoss, self).__init__()
        self.opt = opt

        print('Init Weak Loss, MultiLabelSoftMarginLoss')

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            hm_loss += F.multilabel_soft_margin_loss(output['hm'], batch['cat_id']) / opt.num_stacks

        loss = hm_loss  # zero out wh/off/cam_off loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss}
        return loss, loss_stats

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


class CtCamLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtCamLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()  # default is focalloss
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

        print('Init CtCAM Loss x 0.3')

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        cam_off_loss, cam_aggr_loss = 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss: # 训练时sigmoid, 执行此步骤
                output['hm'] = _sigmoid(output['hm'])
            # FL 需要之前sigmoid，multilabel不需要提前sigmoid

            # use GT to train
            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:  # default False
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                       self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                    batch['dense_wh'] * batch['dense_wh_mask']) /
                                       mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(  # smooth L1 loss
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if 'offset_map' in output:
                # print(batch['offset_mask'].sum(), output['offset_map'].shape, batch['offset_map'].shape)
                # cam_off_loss += (F.l1_loss(output['offset_map'], batch['offset_map'], reduce=False).sum(dim=1)[:,None,] * batch['offset_mask']).mean() / opt.num_stacks
                cam_off_loss += F.l1_loss(output['offset_map'] * batch['offset_mask'],
                                          batch['offset_map'] * batch['offset_mask']) / opt.num_stacks
                # print(output['offset_map'].shape, batch['offset_map'].shape, batch['offset_mask'].shape, batch['offset_map'][0,0], batch['offset_mask'][0])
                # print(cam_off_loss, batch['offset_mask'].sum())
            if 'aggregation' in output:
                cam_aggr_loss += F.multilabel_soft_margin_loss(output['aggregation'], batch['cat_id']) / opt.num_stacks

        if not 'offset_map' in outputs[0]:
            cam_off_loss = torch.tensor([0]).float().cuda()
        if not 'aggregation' in outputs[0]:
            cam_aggr_loss = torch.tensor([0]).float().cuda()

        cam_aggr_loss *= 0.3

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + cam_off_loss + cam_aggr_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss,
                      'cam_off_loss': cam_off_loss, 'cam_aggr_loss': cam_aggr_loss}
        # print(loss_stats)
        return loss, loss_stats

import torch
import numpy as np
import torch.nn.functional as F

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import new_ctdet_decode, pairwise_iou
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


def cal_GTPred_IoU(target, pred):
    b = pred.shape[0]
    iou = []
    for i in range(b):
        subtar = target[target[:,-1]==i, :4] # 最后一维是batch id
        sub_iou = pairwise_iou(pred[i][:,:4], subtar) # 100xN

        # print(pred[i][:,:4].shape, subtar.shape, sub_iou.shape)
        # print(pred[i][:,:4].max(), subtar.max(), sub_iou.max(), sub_iou.sum())
        # print(sub_iou[:10], sub_iou.max(), sub_iou.sum())
        if sub_iou.numel():
            sub_iou = torch.max(sub_iou, dim=-1)[0]
        else:
            sub_iou = torch.zeros(sub_iou.shape[0]).cuda() # [100,0] -> [100]

        iou.append(sub_iou)
    iou = torch.stack(iou, 0)
    return iou # bx100


class CtIoULoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtIoULoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()  # default is focalloss
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

        print('Init Ct IoU Loss x 0.1')

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        cam_off_loss, cam_aggr_loss = 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss: # 训练时sigmoid, 执行此步骤
                output['hm'] = _sigmoid(output['hm'])
            # FL 需要之前sigmoid，multilabel不需要提前sigmoid

            # hm = torch.tensor(anno['hm'], device=output['hm'].device)
            # wh = torch.tensor(anno['wh'], device=output['wh'].device)
            torch.cuda.synchronize()
            dets, ys, xs = new_ctdet_decode(output['hm'], output['wh'], reg=output['reg'], cat_spec_wh=self.opt.cat_spec_wh,
                                K=self.opt.K)  # [1,100,6], x1,y1,x2,y2,conf,cat

            iou_map = torch.zeros_like(output['hm'])
            iou = cal_GTPred_IoU(batch['target_box'], dets).view(-1)
            pos = torch.cat((dets[...,-1].unsqueeze(-1), ys, xs), dim=-1).long()
            b = torch.arange(dets.shape[0]).view(-1, 1, 1).repeat(1, dets.shape[1],1).to(opt.device) # batch id
            pos = torch.cat((b, pos), dim=-1).view(-1, 4)
            print(iou.max(), iou.mean())
            iou_map[tuple(zip(*pos.tolist()))] = iou    # tuple才可以做索引，zip(*list)解包后的每个元素/列表变成tuple

            beta = 0.1
            batch['hm'] += iou_map * beta
            #print(batch['hm'].min(), batch['hm'].max())
            batch['hm'] = batch['hm'].clamp(0,1)

            # with open('%s.txt'%(0), 'w+') as f:
            #     for i in range(batch['hm'][0,0].shape[0]):
            #         for j in range(batch['hm'][0,0].shape[1]):
            #             f.write('%.3f '%batch['hm'][0,0][i][j])
            #         f.write('\n')
            # with open('iou_%s.txt'%(0), 'w+') as f:
            #     for i in range(batch['hm'][0,0].shape[0]):
            #         for j in range(batch['hm'][0,0].shape[1]):
            #             f.write('%.3f '%iou_map[0,0][i][j])
            #         f.write('\n')
            # assert False

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

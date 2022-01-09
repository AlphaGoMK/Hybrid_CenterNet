from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from torchvision import transforms
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math


def save_tensor_image(tensor, filename):
    assert tensor.dim() == 2
    agg = tensor[..., None].repeat(1, 1, 3).detach().cpu()
    agg = (agg + abs(agg.min())) / (agg.max() - agg.min())
    agg = transforms.ToPILImage()(np.uint8(agg.numpy() * 255))
    agg.save(filename, quality=95, subsampling=0)


class CTDetDataset(data.Dataset):

    def _coco_box_to_bbox(self, box):
        # xywh -> xyxy
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def get_by_idx(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        ct_list = []
        ct_mask = torch.zeros(output_h, output_w).long()
        valid_obj = 0
        cat_id = []
        center_ratio = self.opt.center_ratio

        target_box = torch.zeros(0, 4)

        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])

            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            target_box = torch.cat((target_box, torch.tensor(bbox)[None,]), dim=0)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]  # 格子序号
                reg[k] = ct - ct_int  # center offset
                reg_mask[k] = 1  # is target
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

                # valid box
                ct_list.append([ct_int[0], ct_int[1]])
                for i in range(int(max(ct[1] - center_ratio * h / 2, 0)),
                               int(min(ct[1] + center_ratio * h / 2, output_h))):  # y
                    for j in range(int(max(ct[0] - center_ratio * w / 2, 0)),
                                   int(min(ct[0] + center_ratio * w / 2, output_w))):  # x
                        ct_mask[i, j] = 1
                valid_obj += 1
                cat_id.append(cls_id)

        if valid_obj == 0:
            offset_map = torch.zeros(2, output_h, output_w).long()
            cat_target = torch.zeros(num_classes).long()
        else:
            # process offset of center box
            x_dis_map = np.arange(output_w).reshape(1, -1).repeat(output_h, axis=0).reshape(1, output_h,
                                                                                            output_w).repeat(valid_obj,
                                                                                                             axis=0)  # h,w -> obj, h, w
            y_dis_map = np.arange(output_h).reshape(-1, 1).repeat(output_w, axis=1).reshape(1, output_h,
                                                                                            output_w).repeat(valid_obj,
                                                                                                             axis=0)  # h,w -> obj, h, w

            joint_offset = np.concatenate((y_dis_map.reshape(valid_obj, output_h, output_w, 1),
                                           x_dis_map.reshape(valid_obj, output_h, output_w, 1)),
                                          axis=-1)  # obj, h, w, 2
            ct_list = np.array(ct_list).reshape(valid_obj, 1, 1, 2)
            joint_offset -= ct_list[..., [1, 0]]

            distance = abs(joint_offset).sum(axis=-1)

            _, dis_k = torch.tensor(distance).min(dim=0)
            k_idx = dis_k[None, ..., None].repeat(valid_obj, 1, 1, 2)  # expand to offset(same) dimension
            offset_map = torch.gather(torch.tensor(joint_offset), 0, k_idx)[0]  # h, w, 2
            offset_map = offset_map.permute(2, 0, 1)  # 2, h, w

            offset_map = -offset_map  # -1, 1 -> 1, -1

            cat_id = torch.tensor(cat_id).unique()  # 初始化为-1 or 0

            cat_target = torch.zeros(num_classes).long()
            cat_target[cat_id] = 1

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        ret.update(
            {'offset_map': offset_map.numpy(), 'offset_mask': ct_mask[None,].numpy(), 'cat_id': cat_target.numpy()})

        if self.opt.iou_loss:
            ret['target_box'] = target_box

        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta

        return ret

    @staticmethod
    def collate_fn(batch):
        ret = {}
        if 'target_box' in batch[0].keys():
            ret['target_box'] = torch.zeros(0,5)
            for i, r in enumerate(batch):
                new_r = torch.cat((r['target_box'], i*torch.ones(len(r['target_box'])).view(-1,1)), dim=-1)
                ret['target_box'] = torch.cat((ret['target_box'], new_r), dim=0) # x1y1x2y2 img

        for k in batch[0].keys():
            if k == 'target_box':
                continue
            elif k == 'meta':
                ret['meta'] = batch[0]['meta'] # meta仅在eval时出现
            else:
                ret[k] = torch.stack([torch.tensor(batch[i][k]) for i in range(len(batch))], 0)
        return ret

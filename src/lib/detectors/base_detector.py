from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import torch.nn.functional as F
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
import os.path as op
import os, uuid
import random

getfilename = lambda x: op.splitext(op.split(x)[-1])[0]

def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename

class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True
        self.class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                           'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                           'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                           'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                           'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                           'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                           'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.chinese_name = {'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车',
                             'airplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车', 'boat': '船',
                             'traffic light': '红绿灯', 'fire hydrant': '消防栓', 'stop sign': '停止标志',
                             'parking meter': '停车收费表', 'bench': '长凳', 'bird': '鸟', 'cat': '猫', 'dog': '狗',
                             'horse': '马', 'sheep': '羊', 'cow': '牛', 'elephant': '大象', 'bear': '熊',
                             'zebra': '斑马', 'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞',
                             'handbag': '手提包', 'tie': '领带', 'suitcase': '飞盘', 'frisbee': '滑雪板', 'skis': '滑雪板',
                             'snowboard': '单板滑雪', 'sports ball': '运动球', 'kite': '风筝', 'baseball bat': '棒球棒',
                             'baseball glove': '棒球手套', 'skateboard': '滑板', 'surfboard': '冲浪板',
                             'tennis racket': '网球拍', 'bottle': '瓶子 ', 'wine glass': '红酒杯', 'cup': '杯子',
                             'fork': '叉子', 'knife': '刀', 'spoon': '勺子', 'bowl': '碗', 'banana': '香蕉',
                             'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子', 'broccoli': '西兰花',
                             'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '披萨', 'donut': '甜甜圈', 'cake': '蛋糕',
                             'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽', 'bed': '床', 'dining table': '餐桌',
                             'toilet': '马桶', 'tv': '电视', 'laptop': '笔记本电脑', 'mouse': '鼠标',
                             'remote': '遥控器', 'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉',
                             'oven': '烤箱', 'toaster': '烤面包机', 'sink': '洗碗槽', 'refrigerator': '冰箱', 'book': '书',
                             'clock': '时钟', 'vase': '花瓶', 'scissors': '剪刀', 'teddy bear': '泰迪熊',
                             'hair drier': '吹风机', 'toothbrush': '牙刷'}

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False

        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            # 标注
            if self.opt.dataset in ['voc07', 'voc12', 'pascal']:
                anno = {

                }
            else:
                anno = {
                    'hm': image_or_path_or_tensor['hm'][0].numpy(),
                    'wh': image_or_path_or_tensor['wh'][0].numpy(),
                    'reg': image_or_path_or_tensor['reg'][0].numpy(),
                    'offset_map': image_or_path_or_tensor['offset_map'][0].numpy(),
                    'offset_mask': image_or_path_or_tensor['offset_mask'][0].numpy(),
                    'ct_list': image_or_path_or_tensor['ct_list'][0] if len(image_or_path_or_tensor['ct_list']) > 0 else [],
                    'file_name': image_or_path_or_tensor['file_name'][0],
                    'cat_id': image_or_path_or_tensor['cat_id'][0].tolist() if len(
                        image_or_path_or_tensor['cat_id']) > 0 else [],
                    'target_box': image_or_path_or_tensor['target_box'][0].numpy(),
                }
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            if self.opt.debug >= 1:
                output, dets, forward_time = self.process(images, None, return_time=True)
            else:
                output, dets, forward_time = self.process(images, anno, return_time=True)  # 此处执行了sigmoid

            vis_offset = False
            if vis_offset:
                ct_list = torch.tensor(anno['ct_list'])
                if len(ct_list) == 0:
                    ct_list = torch.zeros(1, 2, device=output['hm'].device).long()
                ct_list = ct_list.view(ct_list.shape[0], ct_list.shape[-1])
                off_map = output['offset_map'][0].cpu()
                off_map = (off_map ** 2).sum(dim=0).unsqueeze(0).permute(1, 2, 0)
                torch.set_printoptions(profile="full")
                off_map = (off_map - off_map.min()) / (off_map.max() - off_map.min() + 1e-7)
                off_map = off_map.cpu().numpy()

                off_map[ct_list.T[1], ct_list.T[0]] = 1.
                off_map = cv2.applyColorMap((off_map * 255).astype(np.uint8), cv2.COLORMAP_JET)  # 变channel=3
                off_map = cv2.resize(off_map, (512, 512))
                draw_img = cv2.resize(image, (512, 512))
                fig = (draw_img * 0.5 + off_map * 0.5).astype(np.uint8)
                cv2.imwrite('offset_map/%s' % anno['file_name'], fig)

            vis_hm = False
            if vis_hm:  # 可视化预测/GT的heatmap图
                det_cat = [int(ann_cat) for ann_cat in anno['cat_id']] # 现有类别
                plot_feat = output['hm']
                for cat in det_cat:
                    heatmap = cv2.resize(plot_feat[0, cat][None,].permute(1, 2, 0).cpu().numpy(),
                                         (output['hm'].shape[-1] * 4, output['hm'].shape[-2] * 4))  # param: WH
                    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)  # 变channel=3
                    draw_image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
                    fig = (draw_image * 0.5 + heatmap * 0.5).astype(np.uint8)
                    cv2.imwrite('visual/merge/%s_%s.jpg' % (anno['file_name'].split('.')[0], self.class_name[cat]), fig)
                    print('save visual/merge [%s_%s.jpg]' % (anno['file_name'].split('.')[0], self.class_name[cat]))

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time
            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)

            dets = self.post_process(dets, meta, scale) # 按照类别
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        cam_li = []
        vis_cam = self.opt.vis_cam # 只有在网页端可视化的时候开启
        if vis_cam:
            det_cat = []
            for j in range(1, self.num_classes + 1):
                for bbox in results[j]:
                    if bbox[4] > self.opt.vis_thresh:
                        det_cat.append(j-1)
            det_cat = list(set(det_cat))
            if len(det_cat)>10:
                det_cat = det_cat[:10]

            plot_feat = torch.tensor(output['hm'])

            for cat in det_cat:
                heatmap = cv2.resize(plot_feat[0, cat][None,].permute(1, 2, 0).cpu().numpy(),
                                     (output['hm'].shape[-1] * 4, output['hm'].shape[-2] * 4))  # param: WH
                heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)  # 变channel=3
                draw_image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
                fig = (draw_image * 0.5 + heatmap * 0.5).astype(np.uint8)
                cam_filepath = './static/heatmap/cam_%s.jpg' % self.class_name[cat].replace(' ', '_')
                cv2.imwrite(cam_filepath, fig)
                cam_li.append((self.chinese_name[self.class_name[cat]], cam_filepath.replace('./static/', '')))
                print('save [%s]' % cam_filepath)

        res_img = ''
        if self.opt.debug >= 1: # 都会执行
            if type(image_or_path_or_tensor) == type(''):
                res_img = self.show_results(debugger, image, results, img_name=image_or_path_or_tensor)
            else:
                rand_name = random_filename(''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',7))+'.jpg')
                res_img = self.show_results(debugger, image, results, img_name=rand_name)

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time, 'img_path': res_img, 'cam_li': cam_li}

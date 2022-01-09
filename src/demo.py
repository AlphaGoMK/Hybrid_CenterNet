from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import time
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        total_time = 0.0
        for (image_name) in image_names:
            ret = detector.run(image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            total_time += float(ret['tot'])

        print(total_time/len(image_names))


def det_eval(opt, image_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = 1
    Detector = detector_factory['ctdet']
    detector = Detector(opt)


    if image_name.endswith('.jpg'):
        ret = detector.run(image_name)

        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)

    else:
        start_time = time.time()

        cap = cv2.VideoCapture(image_name)

        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        reso = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ret_path = os.path.join('demo', image_name.split('/')[-1].split('.')[0]+'.mp4')
        ret_vid = cv2.VideoWriter(ret_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, reso)
        if cap.isOpened():
            rval, frame = cap.read()
        else:
            rval = False

        while rval:
            rval, frame = cap.read()
            if not rval:
                break

            ret = detector.run(frame)
            ret_vid.write(cv2.imread(ret['img_path']))

        ret_vid.release()
        print('save at [%s]'%ret_path)
        print('time: [%.3f]ms'%(time.time()-start_time))

        return ret_path, []

    return ret['img_path'], ret['cam_li']


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)

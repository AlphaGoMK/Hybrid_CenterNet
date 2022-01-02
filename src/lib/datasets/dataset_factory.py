from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset

from .dataset.coco import COCO
from .dataset.coco35 import COCO35
from .dataset.coco80 import COCO80
from .dataset.pascal import PascalVOC
from .dataset.voc07 import VOC07
from .dataset.voc12 import VOC12

dataset_factory = {
    'coco': COCO,
    'coco35': COCO35,
    'coco80': COCO80,
    'pascal': PascalVOC,
    'voc07': VOC07,
    'voc12': VOC12,
}

_sample_factory = {
    'ctdet': CTDetDataset,
}

#CTDetDataset = CTDetDataset

def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset

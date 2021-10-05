#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   google_color.py
@Time               :   2021/05/26 23:48:19
@Author             :   Kun Xiong
@Contact            :   xk18@mails.tsinghua.edu.cn
@Last Modified by   :   Kun Xiong
'''

import contextlib
import copy
import datetime
import io
import json
import logging
import os
import os.path as osp

import numpy as np
import torch

from cvpods.structures import Boxes, BoxMode, PolygonMasks
from cvpods.utils import PathManager, Timer, file_lock

from ..base_dataset import BaseDataset
from ..detection_utils import (annotations_to_instances, check_image_size,
                               create_keypoint_hflip_indices,
                               filter_empty_instances, read_image)
from .builtin_meta import _get_builtin_metadata
from .paths_route import _PREDEFINED_SPLITS_COCO

"""
This file contains functions to parse COCO-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger(__name__)


class GoogleColorDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(GoogleColorDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        self.root = "/home/data/dataset/google_colors/"
        self.meta = self._get_metadata()
        self.dataset_dicts = []
        self.color2id = color2id
        import glob
        for path in glob.glob(self.root + '*'):
            if 'README' in path: continue 
            if osp.isdir(path):
                color_name = path.split('/')[-1]
                assert color_name in self.color2id, "Not Exist Color `{}`! Modify the GoogleColor and EbayColor".format(color_name)
                for img_path in glob.glob(osp.join(path,'*')):
                    self.dataset_dicts.append([color_name, img_path])
        
        self.train_dicts = []
        self.test_dicts = []
        for id, (color, path) in enumerate(self.dataset_dicts):
            name_id = int(path.split('/')[-1].split('.')[-2])
            color_id = self.color2id.index(color)
            self.train_dicts.append([color_id, path, id])

        self.dataset_dicts = self.train_dicts
        self._set_group_flag()
        #else : 
        #self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        ret = {}
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # read image
        image = read_image(dataset_dict[1], format=self.data_format)
       # check_image_size(dataset_dict, image)


        # apply transfrom
        image, _ = self._apply_transforms(
            image, None)

        # convert to Instance type
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # h, w, c -> c, h, w
        ret["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        ret["category_id"] = dataset_dict[0]
        ret['document_id'] = dataset_dict[2]
        def load_func(args, load_path):
            return np.load(load_path)

        def save_func(args, data, save_path):
            np.save(save_path, data)

        def proc_func(args):
            return self.calulate_color_bincount(ret['image'].numpy())
            
        from cvpods.utils import DataUnit
        path = osp.join('/home/data/Output/GoogleColor/', self.color2id[dataset_dict[0]] + '__' + osp.basename(dataset_dict[1]).split('.')[-2]) + '.npy'
        dunit = DataUnit(
            'color_word', proc_func, save_func, load_func, path, path, True
        )
        ret["color_word"] = dunit.process()
        from cvpods.utils import DataUnit
        return ret
    
    @staticmethod
    def calulate_color_bincount(image, size=(10,10,10), ctype='rgb'):
        """ calculate the bincount of the 3D ColorCube
            input  : 
                image   : numpy(C, H, W)
            return :      numpy(32 * 32 * 32), the value means the count
        """
        assert (image.shape[0] == 3)
        assert (image.dtype    == np.uint8)
        interval = (255 // size[0] + 1, 255 // size[1] + 1, 255 // size[2] + 1)
        ret = np.zeros((size[0] * size[1] * size[2], ), dtype=np.int32)
        base = [1] + [size[0], size[0]*size[1]]
        for i in range(image.shape[1]): 
            for j in range(image.shape[2]):
                ids = 0
                for c in range(3):
                    pixel = image[c,i,j]
                    ids += pixel // interval[c] * base[c]
                ret[ids] += 1
        return ret

    def __reset__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_dicts)

    def _get_metadata(self):
        meta = {}
        meta["evaluator_type"] = "classification"
        return meta



color2id = [
    'yellow', 
    'white', 
    'red', 
    'purple', 
    'pink',
    'orange', 
    'grey',
    'green',
    'brown',
    'blue',
    'black',
]


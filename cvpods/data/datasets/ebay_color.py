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

class EbayColorDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(EbayColorDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        self.root = "/home/data/dataset/ebay/"
        if not osp.isdir(self.root):    
            raise Exception("You Havn't Install the ebaycolor dataset!")
        self.imagefile = "test_images.txt"
        self.maskfile = "mask_images.txt"

        self.meta = self._get_metadata()
        self.color2id = color2id
        self.category2id = []
        with open(osp.join(self.root, self.imagefile), 'r') as fp : 
            self.image_files = [ _.strip() for _ in fp.readlines()]
        with open(osp.join(self.root, self.maskfile), 'r') as fp : 
            self.mask_files = [ _.strip() for _ in fp.readlines()]

        self.dataset_dicts = []

        for idx, line in enumerate(self.image_files):
            color_name = line.split('/')[-2]
            object_name = line.split('/')[-3]
            if color_name not in self.color2id: assert False, "Not Exist Color `{}`! Modify the GoogleColor and EbayColor".format(color_name)
            ann = {}
            ann['imagepath'] = osp.join(self.root, self.image_files[idx])
            ann['maskpath'] = osp.join(self.root, self.mask_files[idx])
            ann['category_id'] = self.color2id.index(color_name)
            ann['object_name'] = object_name
            ann['document_id'] = idx
            self.dataset_dicts.append(ann)

        self._set_group_flag()
            
        #self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])
        ret = {}
        # read image
        image = read_image(dataset_dict['imagepath'], format=self.data_format)
        mask  = read_image(dataset_dict['maskpath'], format=self.data_format)
        # check_image_size(dataset_dict, image)

        # apply transfrom
        image, _ = self._apply_transforms(
            image, None)
        mask, _ = self._apply_transforms(
            mask , None)

        # convert to Instance type
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # h, w, c -> c, h, w
        ret["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        ret["mask"] = torch.as_tensor(
            np.ascontiguousarray(mask.transpose(2, 0, 1)))
        ret['document_id'] = dataset_dict['document_id']
        ret["category_id"] = dataset_dict["category_id"]
        def load_func(args, load_path):
            return np.load(load_path)

        def save_func(args, data, save_path):
            np.save(save_path, data)

        def proc_func(args):
            return self.calulate_color_bincount(ret['image'].numpy(), ret['mask'].numpy())
            
        from cvpods.utils import DataUnit
        path = osp.join('/home/data/Output/EbayColor/', self.color2id[dataset_dict['category_id']] + '__' + osp.basename(dataset_dict['imagepath']).split('.')[-2]) + '.npy'
        dunit = DataUnit(
            'color_word', proc_func, save_func, load_func, path, path, True
        )
        ret["color_word"] = dunit.process()
        from cvpods.utils import DataUnit
        return ret
    
    @staticmethod
    def calulate_color_bincount(image, mask=None, size=(10,10,10), ctype='rgb'):
        """ calculate the bincount of the 3D ColorCube
            input  : 
                image   : numpy(C, H, W)
            return :      numpy(32 * 32 * 32), the value means the count
        """
        assert (image.shape[0] == 3)
        assert (image.dtype    == np.uint8)
        if mask is None: 
            mask = np.ones_like(image) * 255
        interval = (255 // size[0] + 1, 255 // size[1] + 1, 255 // size[2] + 1)
        ret = np.zeros((size[0] * size[1] * size[2], ), dtype=np.int32)
        base = [1] + [size[0], size[0]*size[1]]
        for i in range(image.shape[1]): 
            for j in range(image.shape[2]):
                if mask[0,i,j] != 255: continue  # mask is set
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

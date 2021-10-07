#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   cifar.py
@Time               :   2021-08-21
@Author             :   Kun Xiong
@Contact            :   
@Last Modified by   :   
@Last Modified time :   
'''

import logging
import os
import os.path as osp
import pickle
import scipy.sparse as sp
import pandas as pd
import random
import torchvision 

import numpy as np
import torch

from copy import deepcopy
from cvpack2.utils import Timer
from cvpack2.data.transforms.transform_gen import TorchTransformGen
from tqdm import tqdm as tq

from ..base_dataset import BaseDataset
from .cifar_utils.generate_noisy_cifar_data import generate_noisy_cifar, read_cifar_100, read_cifar_10
from .paths_route import _PREDEFINED_SPLITS_CIFAR
from ..registry import DATASETS

"""
This file is dataset class of cifar
"""

logger = logging.getLogger(__name__)

@DATASETS.register()
class CIFARDataset(BaseDataset):
    """
    This class contain noise operation, you can specify the noisy ratio of 
    class flip or background flip. 

    You can also set to 0 to enable Raw CIFARDataset.

    cifar_<dataset_name>_<dataset_type>
    dataset_name   [string]     "cifar-10" | "cifar-100"
    dataset_type   [string]     "test" | "noise-val" | "noise-train" | "clean"
    """
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(CIFARDataset, self).__init__(cfg, dataset_name, transforms, is_train)
        self.dataset_name = dataset_name.split('_')[1]
        self.dataset_root = os.path.join(cfg.DATASETS.ROOT, self.dataset_name)
        self.data_type = dataset_name.split('_')[2]
        self.is_train = is_train
        self.validate_number = cfg.DATASETS.VALIDATION_NUM
        self.noise_ratio = cfg.DATASETS.NOISE_RATIO
        self.num_clean = cfg.DATASETS.CLEAN_NUM
        self.with_background = cfg.DATASETS.WITH_BACKGROUND
        self.seed = cfg.SEED
        self.discount = cfg.DATASETS.get("DISCOUNT_RATIO", 1.0)

        self.meta = self._get_metadata()
        self.dataset_dicts, self.dataset_statistic = self._load_annotations()
        self._set_group()

    def _set_group(self):
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, index):
        """ 
        return item of single train sample
        """
        dataset_dict = deepcopy(self.dataset_dicts[index])
        image = dataset_dict['image']
        normalize = torchvision.transforms.Normalize(self.dataset_statistic['mean'], self.dataset_statistic['std'])
        image, _ = self._apply_transforms(image, None)
        image = torch.as_tensor((image.transpose(2, 0, 1))).to(torch.float32)
        image = image * 1.0 / 255.0
        image = normalize(image)
        dataset_dict['image'] = image
        return dataset_dict

    def __len__(self):
        return len(self.dataset_dicts)

    def _get_metadata(self):
        meta = {
            "evaluator_type": _PREDEFINED_SPLITS_CIFAR["evaluator_type"],
        }
        return meta
    
    def _load_annotations(self):
        timer = Timer()
        self.noise_train, self.clean, self.noise_val, self.test, num_classes = generate_noisy_cifar(
            self.dataset_name, self.dataset_root, 
            self.validate_number, self.noise_ratio,
            self.num_clean, self.seed, self.with_background 
            )
        dataset_statistic = {
            'num_classes': num_classes, 
            'format'     : 'HWC' , 
        }
        self.type2data = {
            'noise-train': self.noise_train, 
            'clean'      : self.clean,
            'noise-val'  : self.noise_val, 
            'test'       : self.test, 
        }
        dataset_dicts = []
        current_img, current_label, current_mask = self.type2data[self.data_type]
        if self.dataset_name == 'cifar-10' : 
            dataset_statistic['mean'] = (0.4914, 0.4822, 0.4465)
            dataset_statistic['std'] = (0.2023, 0.1994, 0.2010)
        elif self.dataset_name == 'cifar-100':
            dataset_statistic['mean'] = (0.5071, 0.4867, 0.4408)
            dataset_statistic['std'] = (0.2675, 0.2565, 0.2761)
        else:
            raise RuntimeError("name must be cifar-10 / cifar-100")
        assert (len(current_img) == len(current_label))
        assert (current_mask is None or len(current_img) == len(current_mask))
        #print (current_img[0])
        for i in range(len(current_img)):
            item_dict = {}
            item_dict['image'] = current_img[i] # H x W x C
            item_dict['image_id'] = i # H x W x C
            item_dict['category_id'] = current_label[i] # int
            if current_mask is not None : 
                item_dict['is_clean'] = current_mask[i]
            else :
                item_dict['is_clean'] = 1
            item_dict['width'] = 32
            item_dict['height'] = 32
            item_dict['channel'] = 3
            dataset_dicts.append(item_dict)
        tot_len = len(dataset_dicts)
        dataset_dicts = dataset_dicts[:int(tot_len*self.discount)]
        print  ("dataset length: ", len(dataset_dicts))
        logging.info("Loading CIFAR Images::{} takes {:.2f} seconds.".format(self.dataset_name, timer.seconds()))
        return dataset_dicts, dataset_statistic

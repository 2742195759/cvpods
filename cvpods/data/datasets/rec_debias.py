#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   rec_debias.py
@Time               :   2021-08-21
@Author             :   Kun Xiong
@Contact            :   
@Last Modified by   :   2021-11-10
@Last Modified time :   2021-11-10
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
from cvpods.utils import Timer
from cvpods.data.transforms.transform_gen import TorchTransformGen
from tqdm import tqdm as tq

from ..base_dataset import BaseDataset
from .cifar_utils.generate_noisy_cifar_data import generate_noisy_cifar, read_cifar_100, read_cifar_10
from .paths_route import _PREDEFINED_SPLITS_MOVIELEN
from ..registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register()
class SelectionBiasDataset(BaseDataset):
    """
    Selection Bias Dataset for recommendation.

    movielen_<dataset_name>_<dataset_type>
    dataset_name   [string]     "movielen" | "coat" | "yahoo"
    dataset_type   [string]     "train" | "test" | "clean" | "val"
    """
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(SelectionBiasDataset, self).__init__(cfg, dataset_name, transforms, is_train)
        self.dataset_name = dataset_name.split('_')[1]
        self.dataset_root = cfg.DATASETS.ROOT  # the root of rec_debias
        self.dataset_type = dataset_name.split('_')[2]
        assert self.dataset_name in ["movielen", "coat", "yahoo"], "dataset_name is not implemented"
        assert self.dataset_type in ["train", "test", "clean", "val"], "dataset_type is not implemented"
        self.is_train = is_train
        self.seed = cfg.DATASETS.SEED
        self.with_feature = cfg.DATASETS.get("WITH_FEATURE", False)
        self.meta = self._get_metadata()
        self.user_num = None
        self.item_num = None
        self.dataset_dicts = self._load_annotations()
        self._set_group()

    def _set_group(self):
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, index):
        """ 
        return item of single train sample
        """
        raw = deepcopy(self.dataset_dicts[index])
        dataset_dict = {}
        dataset_dict['user'] = raw[0] - 1
        dataset_dict['item'] = raw[1] - 1
        dataset_dict['score'] = raw[2]
        if self.with_feature: 
            dataset_dict['user_feat'] = self.user_feat[raw[0] - 1]
            dataset_dict['item_feat'] = self.item_feat[raw[1] - 1]

        if self.mat_prop is not None: dataset_dict['propensity'] = self.mat_prop[raw[0]-1, raw[1]-1]
        return dataset_dict

    def __len__(self):
        return len(self.dataset_dicts)

    def _get_metadata(self):
        meta = {
            "evaluator_type": _PREDEFINED_SPLITS_MOVIELEN["evaluator_type"],
        }
        return meta

    def print_counter(self, counter):
        sss = len(list(counter.elements()))
        for k, v in counter.items():
            print ('\t', k, ": ", v*1.0/sss)

    def _load_annotations(self):
        dataset_dicts = self._load_dataset()
        import collections
        self.counter = collections.Counter([ d[2] for d in dataset_dicts ])
        user_set = max([d[0] for d in dataset_dicts])
        item_set = max([d[1] for d in dataset_dicts])
        self.print_counter(self.counter)
        print ("user_num:", (user_set))
        print ("item_num:", (item_set))
        print ("dataset length: ", len(dataset_dicts))
        return dataset_dicts
    
    def _matrix2datasets(self, matrix):
        dataset_dicts = []
        for u in range(self.user_num):
            for i in range(self.item_num):
                s = min(matrix[u, i], 5.0)
                s = max(0,0, s)
                if s > 0.0: dataset_dicts.append((u+1,i+1,s))  # because u, i is 1-based 
        return dataset_dicts

    def _load_feature(self):
        user_path = osp.join(self.dataset_root, self.dataset_name, "user_item_features", "user_features.ascii")
        item_path = osp.join(self.dataset_root, self.dataset_name, "user_item_features", "item_features.ascii")
        self.user_feat = np.loadtxt(user_path)
        self.item_feat = np.loadtxt(item_path)

    def _load_matrix(self, path):
        if 'ascii' in path: matrix = np.loadtxt(path)
        elif 'npy' in path: matrix = np.load(path)
        matrix = matrix.reshape([self.user_num, self.item_num])
        return self._matrix2datasets(matrix)

    def _load_dataset(self):
        timer = Timer()
        if self.with_feature : 
            self._load_feature()
        meta_path = osp.join(self.dataset_root, self.dataset_name, 'meta.txt')
        with open(meta_path, "r") as fp: 
            lines = fp.readlines()
        self.user_num = int(lines[0].strip())
        self.item_num = int(lines[1].strip())

        data_path = osp.join(self.dataset_root, self.dataset_name, self.dataset_type + ".ascii")
        prop_path = osp.join(self.dataset_root, self.dataset_name, "propensities" + ".ascii")
        self.mat_prop = None
        if (osp.exists(prop_path)) : self.mat_prop = np.loadtxt(prop_path)
        dataset_dicts = self._load_matrix(data_path)
        logging.info("Loading dataset ::{} takes {:.2f} seconds.".format(self.dataset_name, timer.seconds()))
        return dataset_dicts

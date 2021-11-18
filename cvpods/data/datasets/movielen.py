#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   movielen.py
@Time               :   2021-10-25
@Author             :   Kun Xiong
@Contact            :   xk18@mails.tinghua.edu.cn
@Last Modified by   :   2021-10-25
@Last Modified time :   2021-10-25
@Discription        :   This file is movielen dataset.
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
class MovieLenDataset(BaseDataset):
    """
    Movie Len Dataset for recommendation.

    movielen_<dataset_name>_<dataset_type>
    dataset_name   [string]     "raw" | "noise" | "clean" | "true"
    dataset_type   [string]     "train" | "test"
    """
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(MovieLenDataset, self).__init__(cfg, dataset_name, transforms, is_train)
        self.dataset_name = dataset_name.split('_')[1]
        self.dataset_root = cfg.DATASETS.ROOT
        self.dataset_type = dataset_name.split('_')[2]
        assert self.dataset_name in ["true", "raw", "noise", "clean"], "Noise is not implemented"
        self.is_train = is_train
        self.num_fold = 1 # [1, 5]
        self.seed = cfg.DATASETS.SEED
        self.alpha = cfg.DATASETS.get("ALPHA", 0.25)
        self.meta = self._get_metadata()
        self.user_num = 943
        self.item_num = 1682
        self.clean_num = cfg.DATASETS.get("CLEAN_NUM", 1000)
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
        dataset_dict['user'] = raw[0]
        dataset_dict['item'] = raw[1]
        dataset_dict['score'] = raw[2]
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
        function_name = "_load_" + self.dataset_name
        print ("start: %s" % function_name)

        dataset_dicts = getattr(self, function_name)()
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
        

    def _load_matrix(self, path):
        if 'ascii' in path: matrix = np.loadtxt(path)
        elif 'npy' in path: matrix = np.load(path)
        matrix = matrix.reshape([self.user_num, self.item_num])
        return self._matrix2datasets(matrix)

    def _load_noise(self):
        timer = Timer()
        if self.dataset_type == "train":
            path = osp.join(self.dataset_root, "select_bias", "observe_matrix_" + str(self.alpha) + ".npy")
        else : 
            path = osp.join(self.dataset_root, "select_bias", "clean_matrix.npy")
        dataset_dicts = self._load_matrix(path)
        logging.info("Loading MovieLen ::{} takes {:.2f} seconds.".format("movie_len_noise", timer.seconds()))
        return dataset_dicts

    def _load_true(self):
        timer = Timer()
        path = osp.join(self.dataset_root, "select_bias", "matrix_after" + ".npy")
        dataset_dicts = self._load_matrix(path)
        logging.info("Loading MovieLen ::{} takes {:.2f} seconds.".format("movie_len_true", timer.seconds()))
        return dataset_dicts

    def _load_clean(self):
        timer = Timer()

        def sample_clean(matrix, clean_num):
            mat = matrix.reshape([-1])
            clean_mat = np.zeros_like(mat)
            import random
            random.seed(self.seed)
            pool= set()
            while(len(pool) != clean_num):
                ind = random.choice(range(mat.size))
                if ind not in pool: 
                    clean_mat[ind] = mat[ind]
                    pool.add(ind)
            print  (list(pool)[:10])
            assert (clean_mat != 0).sum() == clean_num, "Error"
            return clean_mat

        if self.dataset_type == "train":
            path = osp.join(self.dataset_root, "select_bias", "matrix_after.npy")
            matrix = np.load(path)
            matrix = sample_clean(matrix, self.clean_num)
            matrix = matrix.reshape([self.user_num, self.item_num])
            dataset_dicts = self._matrix2datasets(matrix)
        else : 
            path = osp.join(self.dataset_root, "select_bias", "clean_matrix.npy")
            dataset_dicts = self._load_matrix(path)

        logging.info("Loading MovieLen ::{} takes {:.2f} seconds.".format("movie_len_noise", timer.seconds()))
        return dataset_dicts
        

    def _load_raw(self):
        timer = Timer()
        mmap = {
            "train": "base", 
            "test" : "test",
        }
        path = osp.join(self.dataset_root, "u" + str(self.num_fold) + "." + mmap[self.dataset_type])
        with open(path, "r") as fp :
            lines = fp.readlines()

        dataset_dicts = []

        for line in lines:
            fields = line.strip().split("\t")
            user = int(fields[0])
            item = int(fields[1])
            score = int(fields[2])
            dataset_dicts.append((user,item,score))

        logging.info("Loading MovieLen ::{} takes {:.2f} seconds.".format(self.dataset_name, timer.seconds()))
        return dataset_dicts


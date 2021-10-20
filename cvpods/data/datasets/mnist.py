#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   mnist.py
@Time               :   2021-08-21
@Author             :   Kun Xiong
@Contact            :   
@Last Modified by   :   2021-10-18
@Last Modified time :   2021-10-18
@Discription        :   This file is dataset of mnist. contain the imbalance option in learning to reweight.
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
from .paths_route import _PREDEFINED_SPLITS_CIFAR
from ..registry import DATASETS


def get_imbalance_dataset(mnist_train,
                          mnist_test ,
                          pos_ratio=0.9,
                          ntrain=5000,
                          nval=10,
                          ntest=500,
                          seed=0,
                          class_0=4,
                          class_1=9):
    rnd = np.random.RandomState(seed)

    # In training, we have 10% 4 and 90% 9.
    # In testing, we have 50% 4 and 50% 9.
    ratio = 1 - pos_ratio
    ratio_test = 0.5

    x_train = np.stack([d[0].numpy() for d in mnist_train], axis=0)
    y_train = np.array([d[1] for d in mnist_train])
    x_test = np.stack([d[0].numpy() for d in mnist_test], axis=0)
    y_test = np.array([d[1] for d in mnist_test])

    x_train_0 = x_train[y_train == class_0]
    x_test_0 = x_test[y_test == class_0]

    # First shuffle, negative.
    idx = np.arange(x_train_0.shape[0])
    rnd.shuffle(idx)
    x_train_0 = x_train_0[idx]

    nval_small_neg = int(np.floor(nval * ratio_test))
    ntrain_small_neg = int(np.floor(ntrain * ratio)) - nval_small_neg

    x_val_0 = x_train_0[:nval_small_neg]    # 450 4 in validation.
    x_train_0 = x_train_0[nval_small_neg:nval_small_neg + ntrain_small_neg]    # 500 4 in training.

    if True:
        print('Number of train negative classes', ntrain_small_neg)
        print('Number of val negative classes', nval_small_neg)

    idx = np.arange(x_test_0.shape[0])
    rnd.shuffle(idx)
    x_test_0 = x_test_0[:int(np.floor(ntest * ratio_test))]    # 450 4 in testing.

    x_train_1 = x_train[y_train == class_1]
    x_test_1 = x_test[y_test == class_1]

    # First shuffle, positive.
    idx = np.arange(x_train_1.shape[0])
    rnd.shuffle(idx)
    x_train_1 = x_train_1[idx]

    nvalsmall_pos = int(np.floor(nval * (1 - ratio_test)))
    ntrainsmall_pos = int(np.floor(ntrain * (1 - ratio))) - nvalsmall_pos

    x_val_1 = x_train_1[:nvalsmall_pos]    # 50 9 in validation.
    x_train_1 = x_train_1[nvalsmall_pos:nvalsmall_pos + ntrainsmall_pos]    # 4500 9 in training.

    idx = np.arange(x_test_1.shape[0])
    rnd.shuffle(idx)
    x_test_1 = x_test_1[idx]
    x_test_1 = x_test_1[:int(np.floor(ntest * (1 - ratio_test)))]    # 500 9 in testing.

    if True: 
        print('Number of train positive classes', ntrainsmall_pos)
        print('Number of val positive classes', nvalsmall_pos)

    y_train_subset = np.concatenate([np.zeros([x_train_0.shape[0]]), np.ones([x_train_1.shape[0]])])
    y_val_subset = np.concatenate([np.zeros([x_val_0.shape[0]]), np.ones([x_val_1.shape[0]])])
    y_test_subset = np.concatenate([np.zeros([x_test_0.shape[0]]), np.ones([x_test_1.shape[0]])])

    y_train_pos_subset = np.ones([x_train_1.shape[0]])
    y_train_neg_subset = np.zeros([x_train_0.shape[0]])

    x_train_subset = np.concatenate([x_train_0, x_train_1], axis=0).reshape([-1, 28, 28, 1])
    x_val_subset = np.concatenate([x_val_0, x_val_1], axis=0).reshape([-1, 28, 28, 1])
    x_test_subset = np.concatenate([x_test_0, x_test_1], axis=0).reshape([-1, 28, 28, 1])

    x_train_pos_subset = x_train_1.reshape([-1, 28, 28, 1])
    x_train_neg_subset = x_train_0.reshape([-1, 28, 28, 1])

    # Final shuffle.
    idx = np.arange(x_train_subset.shape[0])
    rnd.shuffle(idx)
    x_train_subset = x_train_subset[idx]
    y_train_subset = y_train_subset[idx]

    idx = np.arange(x_val_subset.shape[0])
    rnd.shuffle(idx)
    x_val_subset = x_val_subset[idx]
    y_val_subset = y_val_subset[idx]

    idx = np.arange(x_test_subset.shape[0])
    rnd.shuffle(idx)
    x_test_subset = x_test_subset[idx]
    y_test_subset = y_test_subset[idx]

    train_set = (x_train_subset * 255.0, y_train_subset)
    train_pos_set = (x_train_pos_subset * 255.0, y_train_pos_subset)
    train_neg_set = (x_train_neg_subset * 255.0, y_train_neg_subset)
    val_set = (x_val_subset * 255.0, y_val_subset)
    test_set = (x_test_subset * 255.0, y_test_subset)

    return train_set, val_set, test_set, train_pos_set, train_neg_set

def get_torch_mnist_dataset():
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    train = torchvision.datasets.MNIST('/home/data/dataset/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    test = torchvision.datasets.MNIST('/home/data/dataset/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    return train, test

logger = logging.getLogger(__name__)

@DATASETS.register()
class MNISTDataset(BaseDataset):
    """
    This class contain noise operation, you can specify the noisy ratio of 
    class flip or background flip. 

    You can also set to 0 to enable Raw CIFARDataset.

    mnist_<dataset_name>_<dataset_type>
    dataset_name   [string]     "raw" | "imb"
    dataset_type   [string]     "train" | "test" | "val"
    """
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(MNISTDataset, self).__init__(cfg, dataset_name, transforms, is_train)
        self.dataset_name = dataset_name.split('_')[1]
        self.dataset_root = os.path.join(cfg.DATASETS.ROOT, self.dataset_name)
        self.data_type = dataset_name.split('_')[2]
        self.is_train = is_train
        self.validate_number = cfg.DATASETS.VALIDATION_NUM
        self.noise_ratio = cfg.DATASETS.NOISE_RATIO
        self.num_clean = cfg.DATASETS.CLEAN_NUM
        self.seed = cfg.SEED

        self.meta = self._get_metadata()
        self._load_annotations()
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
        raw_trainset, raw_testset = get_torch_mnist_dataset()
        # every set is a tuple of (datas, labels)
        __import__('pdb').set_trace()
        train_set, val_set, test_set, train_pos_set, train_neg_set = get_imbalance_dataset(raw_trainset, raw_testset) 

        dataset_statistic = {
            'num_classes': 10, 
            'format'     : 'HWC' , 
        }
        self.type2data = {
            'train'      : train_set, 
            'val'        : val_set,
            'test'       : test_set, 
        }
        __import__('pdb').set_trace()

        dataset_dicts = []
        current_img, current_label, current_mask = self.type2data[self.data_type]
        assert (len(current_img) == len(current_label))
        assert (current_mask is None or len(current_img) == len(current_mask))
        for i in range(len(current_img)):
            item_dict = {}
            item_dict['image'] = current_img[i] # H x W x C
            item_dict['image_id'] = i # H x W x C
            item_dict['category_id'] = current_label[i] # int
            if current_mask is not None : 
                item_dict['is_clean'] = current_mask[i]
            else :
                item_dict['is_clean'] = 1
            item_dict['width'] = 28
            item_dict['height'] = 28
            item_dict['channel'] = 1
            dataset_dicts.append(item_dict)
        tot_len = len(dataset_dicts)
        dataset_dicts = dataset_dicts[:int(tot_len*self.discount)]
        print  ("dataset length: ", len(dataset_dicts))
        logging.info("Loading CIFAR Images::{} takes {:.2f} seconds.".format(self.dataset_name, timer.seconds()))
        return dataset_dicts, dataset_statistic

if __name__ == "__main__":
    dataset = get_torch_mnist_dataset()
    get_imbalance_dataset(dataset)
    

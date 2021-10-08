#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   rec_amazon.py
@Time               :   
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

import numpy as np
import torch

from copy import deepcopy
from cvpods.utils import Timer
from tqdm import tqdm as tq

from ..registry import DATASETS
from ..base_dataset import BaseDataset
from .paths_route import _PREDEFINED_SPLITS_AMAZON

"""
This file contains functions to parse AmazonDataset to cvpods dataset
"""

logger = logging.getLogger(__name__)

class DataLoader():
    def __init__(self, path, efm=False):
        """ if with efm , the return will have features
        """
        print("loading data ...")
        print("efm model = ", efm)
        self._efm = efm
        np.random.seed(0)
        random.seed(0)
        if path[-1] != '/': path += '/'
        self.path = path
        self.statistics = dict()
        user_id_dict = pickle.load(open(self.path + "user_id_dict","rb"))
        item_id_dict = pickle.load(open(self.path + "item_id_dict", "rb"))
        feature_id_dict = pickle.load(open(self.path + "feature_id_dict", "rb"))
        id_feature_dict = pickle.load(open(self.path + "id_feature_dict", "rb"))
        self.statistics['user_number'] = len(user_id_dict.keys())
        self.statistics['item_number'] = len(item_id_dict.keys())
        self.statistics['feature_number'] = len(feature_id_dict.keys())

        self.train_batch_data = pd.read_csv(self.path + "train_data", header=None, dtype='str')

        self.train_user_positive_items_dict = pickle.load(open(self.path + "train_user_positive_items_dict", "rb"))
        self.train_user_negative_items_dict = pickle.load(open(self.path + "train_user_negative_items_dict", "rb"))

        self.ground_truth_user_items_dict = pickle.load(open(self.path + "test_ground_truth_user_items_dict", "rb"))
        self.compute_user_items_dict = pickle.load(open(self.path + "test_compute_user_items_dict", "rb"))

        #print (self.ground_truth_user_items_dict)
        #print (self.compute_user_items_dict )
            
        print(self.statistics)

        self.user_all = []
        self.user_feature_all = []
        self.pos_item_all = []
        self.neg_item_all = []
        self.label_all = []
        if self._efm:
            self.x_uf = []
            self.x_if = []
            self.ui2feature = {}
            self.pos_feature_all = []
            self._calculate_x_y_score()
            
            print ("Interaction: ", len(self.ui2feature))
            print ("Density: ", len(self.ui2feature) * 1.0 / (self.statistics['user_number'] * self.statistics['item_number'] * 1.0))
            
            # Count the most frequently mentioned features
            fid2freq = {}
            for key, feats in self.ui2feature.items():
                for fid in feats:
                    fid2freq[fid] = fid2freq.get(fid, 0) + 1
            most_k = 10 
            fid_most = sorted(fid2freq.items(), key=lambda x: x[1], reverse=True)[:most_k]
            for fid, freq in fid_most:
                print ("Most Feature:", id_feature_dict[fid], freq)

            
    def _calculate_x_y_score(self):
        X = {}
        Y = {}
        user_feature_number = {}
        item_feature_number = {}
        train_data = self.train_batch_data.values.tolist()
        for single_item in train_data:
            user, item, f, s = int(single_item [0]), int(single_item [1]), int(single_item [2]), int(single_item [3])
            feature_key  = str(user)+'@'+str(item)
            self.ui2feature[feature_key] = self.ui2feature.get(feature_key, []) + [f]

            if user not in user_feature_number:
                user_feature_number[user] = {}
                user_feature_number[user][f] = 1
            else:
                if f not in user_feature_number[user]:
                    user_feature_number[user][f] = 1
                else:
                    user_feature_number[user][f] += 1

            if item not in item_feature_number:
                item_feature_number[item] = {}
                item_feature_number[item][f] = int(s)
            else:
                if f not in item_feature_number[item]:
                    item_feature_number[item][f] = int(s)
                else:
                    item_feature_number[item][f] += int(s)

        print('----------------------')# CHECK OK
        for user, features in user_feature_number.items():
            for feature, number in features.items():
                k = str(user) + '@' + str(feature)
                score  = 1 + 4 * (2/(1+np.exp(-number)) -1)
                X[k] = score
        print('----------------------')# CHECK OK
        for item, features in item_feature_number.items():
            for feature, number in features.items():
                k = str(item) + '@' + str(feature)
                score = 1 + 4 / (1 + np.exp(-number))
                Y[k] = score
        print('----------------------')
        self.X = X 
        self.Y = Y 

    def generate_pair_wise_training_corpus(self):
        for user, positive_items in self.train_user_positive_items_dict.items():
            if user in self.train_user_negative_items_dict.keys():
                for item in positive_items:
                    pos_item_id = int(item)
                    neg_item_id = random.choice(self.train_user_negative_items_dict[user])
                    
                    if not self._efm:
                        self.user_all.append(user)   # 用户的id
                        self.pos_item_all.append(pos_item_id)   # 正的item id
                        self.neg_item_all.append(neg_item_id)  #  负的item id
                    if self._efm: 
                        if str(user)+'@'+str(item) not in self.ui2feature:
                            logger.warn("user@pos_item key not in training set! Dataset may have some wrong")
                            continue
                        for pos_feat in self.ui2feature[str(user)+'@'+str(item)]:
                            self.x_uf.append(self.X[str(user)+'@'+str(pos_feat)])
                            self.x_if.append(self.Y[str(item)+'@'+str(pos_feat)])
                            self.pos_feature_all.append(pos_feat)
                            self.user_all.append(user)   # 用户的id
                            self.pos_item_all.append(pos_item_id)   # 正的item id
                            self.neg_item_all.append(neg_item_id)  #  负的item id

@DATASETS.register()
class AmazonDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(AmazonDataset, self).__init__(cfg, dataset_name, transforms, is_train)
        self.with_feature = cfg.DATASETS.WITH_FEATURE & is_train # if test mode, then must be not with feature
        print ("with feature = ", self.with_feature)
        self.dataset_root = cfg.DATASETS.ROOT
        self.dataset_name = "_".join(dataset_name.split('_')[1:])
        self.is_train = is_train

        self.meta = self._get_metadata()
        self.dataset_dicts, self.dataset_statistic = self._load_annotations()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)
        self._set_group()

        print ('dataset length:', len(self))

    def _set_group(self):
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, index):
        """ 
        return item of single train sample
        if is_train: 
            return  dict(
                "user": int64
                "pos_item": int64
                "neg_item": int64
            )

        if not is_train:
            return  dict (
                "user":     int64
                "candidate"  : list[100]   (
                            int64
                        )
                "gt"         : list(
                            int64
                        )
            )
        """
        dataset_dict = deepcopy(self.dataset_dicts[index])
        return dataset_dict

    def __len__(self):
        return len(self.dataset_dicts)

    def _get_metadata(self):
        meta = {
            "evaluator_type": _PREDEFINED_SPLITS_AMAZON["evaluator_type"],
        }
        return meta
    

    def _load_annotations(self):
        timer = Timer()
        dataloader = self.dataloader = DataLoader(osp.join(self.dataset_root, self.dataset_name), self.with_feature)
        dataloader.generate_pair_wise_training_corpus()
        logging.info("Loading AmazonDataset::{} takes {:.2f} seconds.".format(self.dataset_name, timer.seconds()))
        dataset_dicts = []
        dataset_statistic = dataloader.statistics

        if (self.is_train): # fill the dataset_dicts
            if (not self.with_feature):
                for user, pos_item, neg_item in zip(dataloader.user_all,
                                   dataloader.pos_item_all,  
                                   dataloader.neg_item_all, ):
                    dataset_dicts.append({
                        "user": user,
                        #"feat": feat, 
                        "pos_item": pos_item, 
                        #"pos_feat": pos_feat, 
                        "neg_item": neg_item, 
                        #"neg_feat": neg_feat, 
                    })
            elif (self.with_feature):
                for user, pos_item, neg_item, feat, x_uf, x_if in zip(dataloader.user_all,
                                   dataloader.pos_item_all,  
                                   dataloader.neg_item_all, 
                                   dataloader.pos_feature_all, dataloader.x_uf, dataloader.x_if
                                   ):
                    dataset_dicts.append({
                        "user": user,
                        "feat": feat, 
                        "pos_item": pos_item, 
                        "neg_item": neg_item, 
                        "x_uf": x_uf,
                        "x_if": x_if
                    })
                

        else:           # fill the dataset_dicts with different logic
            #assert len(dataloader.ground_truth_user_items_dict) == len(dataloader.compute_user_items_dict), "AmazonDataset have some wrong with it, len(gt)!=len(candidate)"
            validate_user = set(dataloader.ground_truth_user_items_dict.keys()) & set(dataloader.compute_user_items_dict.keys())
            for user in validate_user:
                gt = dataloader.ground_truth_user_items_dict[user]
                candidate = dataloader.compute_user_items_dict[user]
                dataset_dicts.append({
                    "user": user, 
                    "gt"  : gt,
                    "candidate": candidate, 
                })

        return dataset_dicts, dataset_statistic

@DATASETS.register()
class AmazonDatasetSubstitution(AmazonDataset):
    def __init__(self, cfg, dataset_name, transforms, is_train=True):
        super(AmazonDatasetSubstitution, self).__init__(cfg, dataset_name, transforms, is_train)

        assert self.with_feature == True or self.is_train == False, "AmazonDatasetSubstitution must have cfg.DATASETS.WITH_FEATURE = True, but False is found"
        # self.amazon_datset = AmazonDataset(cfg, dataset_name, transforms, is_train)
        self._lazy = cfg.DATASETS.LAZY_SAMPLE
        self._popularity_biased_sampling()
    
    #TODO 可以将这个过程融入到训练中，这样不会有很长的预处理时间，但是会增加总训练长度（或许使用多线程不会。可以将这些过程作为一个 global_transform 来实现）
    def _popularity_biased_sampling (self):
        """ 使用采样方法，改变self.data_dicts的每个item，将每个item添加 item_query
        """
        assert hasattr(self, "dataloader"), "AmazonDataset don't have self.dataloader"
        dataloader = self.dataloader

        train_data = dataloader.train_batch_data.values.tolist()
        items_pop = {}
        user2items = {}
        for single_item in train_data:
            user, item, f, s = int(single_item [0]), int(single_item [1]), int(single_item [2]), int(single_item [3])
            items_pop[item] = items_pop.get(item, 0) + 1
            user2items[user] = user2items.get(user, []) + [item]
        user2items = {k:set(v) for k, v in user2items.items()}
        import math
        import numpy as np;
        items_pop = {k:math.pow(v, 0.75) for k,v in items_pop.items()}

        self.items_pop = items_pop
        self.user2items = user2items

        if not self._lazy : 
            ret = []
            print ("Substitution Sampling Process: ")
            for item in tq(self.dataset_dicts):
                ret.append(foreach_sample(item))
            #self.dataset_dicts = [ foreach_sample(item) for item in self.dataset_dicts ]
            self.dataset_dicts = ret

    def sample(self, userid):
        tmp = {k:v for k,v in self.items_pop.items() if k not in self.user2items[userid]}
        tot = sum(tmp.values())
        keys = np.array([ k for k,v in tmp.items() ])
        vals = np.array([ 1.0*v/tot for k,v in tmp.items() ])
        ret = np.nonzero(np.random.multinomial(1, vals))
        return keys[ret][0]
    
    def foreach_sample(self, dataset_dict): # 没有闭包，可以
        import copy
        tmp = copy.deepcopy(dataset_dict)
        tmp['item_query'] = self.sample(tmp['user'])
        return tmp
        
    def __getitem__(self, index):
        """ 
        return item of single train sample
        if is_train: 
            return  dict(
                "user": int64
                "pos_item": int64
                "neg_item": int64
            )

        if not is_train:
            return  dict (
                "user":     int64
                "candidate"  : list[100]   (
                            int64
                        )
                "gt"         : list(
                            int64
                        )
            )
        """
        dataset_dict = deepcopy(self.dataset_dicts[index])
        if self._lazy : 
            return self.foreach_sample(dataset_dict)
        else : 
            return dataset_dict

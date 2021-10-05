import tqdm#{{{
import json
import pdb
import logging
import copy
import os
import os.path as osp
import random

import numpy as np
import torch
from easydict import EasyDict as edict

from copy import deepcopy
from cvpack2.utils import Timer
import cvpack2

from ..base_dataset import BaseDataset
from ..detection_utils import annotations_to_instances, filter_empty_instances, read_image
from .paths_route import _PREDEFINED_SPLITS_REFERIT
from cvpack2.structures import Boxes, Instances, BoxMode
from cvpack2.data.transforms import IoUCropTransform, ScaleTransform
from cvpack2.evaluation.proposal_evaluation import ProposalLoader
import os
import os.path as osp
import pickle
import os
from cvpack2.utils.distributed import comm
import h5py
#}}}

#] FIXME : bugs here, if this code put in __init__, then multi-gpu will get : refer module not found
refer_tool_root = "/home/data/dataset/cv/referit/refer/"
refer = cvpack2.utils.imports.dynamic_import('refer', refer_tool_root)
image_dir = "/home/data/dataset/cv/referit/refer/data/images/mscoco/train2014/"
color_word_dir = "/home/data/Output/ReferItColorWord/"
"""
special tokens : 
<eos>  <sos>  unknow
"""

class WrapTensor:#{{{
    def __init__(self, data):
        self.data = data
        ...

class SharedTensor(object):
    def __init__(self):
        self.data = None
        ...

    def copy_(self, src_tnr):
        if self.data is None:
            self.data = src_tnr
            self.data = self.data.share_memory_()
        else : 
            self.data.copy_(src_tnr)
        return self.data
#}}}
class DatasetTransformContext(object):#{{{
    def __init__(self):
        self._context_dict = {}
        pass

    def __setitem__ (self, key, val):
        assert type(key) == str, "the key must be string in DatasetTransformContext"
        if key in self._context_dict:
            assert False, "The information str is seted, change another key"
        self._context_dict[key] = val
        
    def __getitem__ (self, key):
        assert key in self._context_dict, "The key your tranform need is not in global context {}".format(key)
        return self._context_dict[key]
#}}}
class DatasetTransform(object):#{{{
    def __init__(self, runtime=False):
        self._runtime = runtime
        pass 
    def pre_process (self, dds, context):
        pass
    def post_process(self, dds, context):
        pass
    def process(self, dd, context):
        """ 处理一个item，并且返回
        """
        pass
    def _get_by_search_key(self, dd, search_key):
        """ util function, search the val and father by search_key
        """
        assert (type(search_key) == list)
        if len(search_key) == 0 : return None
        tmp = dd
        for key in search_key:
            assert key in tmp, "Can fetch val by your search_key: " + key
            tmp = tmp[key]
        return tmp

    def _set_by_search_key(self, dd, new_val, search_key):
        assert (type(search_key) == list)
        if len(search_key) == 0 : return None
        tmp = dd
        for key in search_key[:-1]:
            if key not in tmp:
                tmp[key] = {}
            tmp = tmp[key]
        tmp[search_key[-1]] = new_val
#}}}
class DatasetTransform_Unknowlize_Truncate(DatasetTransform):#{{{
    def __init__(self, runtime, freq_threshold=-1, max_len=15, tokens_key=['tokens'], store_key=["tmp_tokens"]):
        super(DatasetTransform_Unknowlize_Truncate, self).__init__(runtime)
        self._vocabulary = {}
        self._ft = freq_threshold
        self._tokens_key = tokens_key
        self._store_key = store_key
        self._max_len = max_len

    def pre_process (self, dds, context):
        " ================= 计算 单词频率 ==================="
        voc_freq = {}
        for dd in dds : 
            tokens = self._get_by_search_key(dd, self._tokens_key)
            assert (type(tokens) == list)
            for token in tokens:
                token = token.lower()
                if token not in voc_freq: voc_freq[token] = 0
                voc_freq[token] += 1
        context['voc_freq'] = voc_freq
        self._voc_freq = voc_freq
        #invalid_cnt = len([ 1 for k,v in voc_freq.items() if v < filter_freq ])
        #tot_cnt     = len(voc_freq)
        #print ('invalid rate :', invalid_cnt*1.0 / tot_cnt)
        #print ('voc_size     :', tot_cnt - invalid_cnt)

    def process(self, dd, context):
        " =============== 替换tokens和raw ==================="
        filter_freq = self._ft
        tokens = self._get_by_search_key(dd, self._tokens_key)
        new_tokens = ['<SOS>']
        for token in tokens[:self._max_len-2] : 
            token = token.lower()
            assert( token in self._voc_freq )
            if self._voc_freq[token] < filter_freq : 
                new_tokens.append('unknow')
                #return None  # remove the low frequence words
            else : new_tokens.append(token)
        new_tokens.append("<EOS>")
        if len(new_tokens) <= 2: return None
        self._set_by_search_key(dd, new_tokens, self._store_key)
        return dd
#}}}
class DatasetTransform_Tokenize(DatasetTransform):#{{{
    """ 这个Transform将数据集中的token变化为id, 并且填充为同一个类型
        必须先进行

        behavior : 
            if is_train: then gather all the token and gather them as a vocabulary
            if not is_train: then use the saved vocabulary

            so the vocabulary is always the same, and constructed from training set
            if token from test set not appear in training set, use unknow instead
    """
    def __init__(self, runtime=False, padding=True, tokens_key=['tmp_tokens'], store_key=['token_ids'], is_train=True, saved_file='vocabulary.pkl'):
        """ is_train 
        """
        super(DatasetTransform_Tokenize, self).__init__(runtime)
        self._word2id = {'unknow': 0}
        self._id2word = {0: 'unknow'}
        self._tokens_key = tokens_key
        self._store_key = store_key
        self._padding = padding
        self._max_len = 0
        self._save_file = saved_file
        self._is_train = is_train
    
    def pre_process(self, dds, context):
            
        word2id = self._word2id 
        id2word = self._id2word
        if not self._is_train:
            print ('loading vocabulary from {}'.format(self._save_file))
            with open(self._save_file, 'rb') as fp:
                word2id, id2word = pickle.load(fp) 
        else : 
            cnt_id = 1
            for dd in dds:
                tokens = self._get_by_search_key(dd, self._tokens_key)
                self._max_len = max(self._max_len, len(tokens))
                for token in tokens :
                    token = token.lower()
                    if token not in word2id: 
                        word2id[token] = cnt_id
                        id2word[cnt_id]= token
                        cnt_id += 1
            print ('saving vocabulary into {}'.format(self._save_file))
            with open(self._save_file, 'wb') as fp:
                pickle.dump([word2id, id2word], fp)

        context['word2id'] = word2id
        context['id2word'] = id2word
        self._word2id = word2id
        self._id2word = id2word

    def process(self, dd, context):
        unknow_id = self._word2id['unknow']
        tokens = self._get_by_search_key(dd, self._tokens_key)
        token_ids = [self._word2id.get(token.lower(), unknow_id) for token in tokens]
        pad_size = self._max_len - len(token_ids)
        token_ids = token_ids + [self._word2id['<EOS>'.lower()] for _ in range(pad_size)]
        self._set_by_search_key(dd, token_ids, self._store_key)
        return dd#}}}
class DatasetTransform_ExcludeSmallGt(DatasetTransform):#{{{
    def __init__(self, thres_area=10000, runtime=False):
        super(DatasetTransform_ExcludeSmallGt, self).__init__(runtime)
        self._thres_area = thres_area
    
    def process(self, dd, context):
        gt = dd['bbox']
        area = gt[2] * gt[3]
        if area > self._thres_area: return dd
        return None#}}}
class DatasetTransform_ExcludeEmptyProposal(DatasetTransform):#{{{
    def __init__(self, runtime=False):
        super(DatasetTransform_ExcludeEmptyProposal, self).__init__(runtime)
    def process(self, dd, context):
        if context['proposal_dict'] is None: return dd # don't use the proposal feat
        file_name = dd['file_name']
        if file_name not in context['proposal_dict'] or len(context['proposal_dict'][file_name]['proposals']) == 0: 
            #if file_name not in context['proposal_dict']:
                #print ("[WARN] FILE NOT FOUND")
            return None 
        else :
            return dd   #}}}
class DatasetTransform_AddSpatialFeature(DatasetTransform):#{{{
    def __init__(self, runtime=True):
        """ must in runtime mode, because in this time , we have annotations
            add the spatial features in the given 'gt', 'prop', 'neg' type annotations
        """
        super(DatasetTransform_AddSpatialFeature,self).__init__(runtime)
    @staticmethod
    def _cal_spatial_feature(box_array, width, height):#{{{
        """ box_array: n x 4
            return   : n x 5 # spatials features
        """
        ori_props = box_array.reshape([-1, 4])
        ori_props[:,0] = ori_props[:,0] / width 
        ori_props[:,2] = ori_props[:,2] / width 
        ori_props[:,1] = ori_props[:,1] / height 
        ori_props[:,3] = ori_props[:,3] / height #  N x 4
        area_props     = (ori_props[:,2] - ori_props[:,0]) * (ori_props[:,3] - ori_props[:,1])
        area_props     = np.reshape(area_props, [-1,1])
        spatial_feats  = np.concatenate([ori_props, area_props], axis=1)
        return spatial_feats#}}}
    def _ann_to_spatial(self, dd, anns: list, ann_type: str, ins_key: str):
        """ collect the ann_type as list and calculate spatials, then attach to dd['<ins_key>'] Instances
        """
        height, width = dd['image_size']
        box_list = [ ann['origin_bbox'] for ann in anns if ann['type'] == ann_type ]
        if (len(box_list) == 0): 
            dd[ins_key].spatials = torch.Tensor([]).reshape([0, 5]).float()
            return 
        ori_props = np.array(box_list)
        dd[ins_key].spatials = torch.Tensor(self._cal_spatial_feature(ori_props, width, height))

    def process(self, dd, context):
        """ add spatial feature for every proposals

            need : dd['annotations']
            set  : dd['props_ins'].proposal_spatial = Tensor (N x 5)
        """
        anns = dd['annotations']
        type_set = ['gt', 'prop', 'neg']  #set([ ann['type'] for ann in dd['annotations'] ])
        for type_str in type_set:
            out_str = type_str + 's_ins' if type_str != 'gt' else type_str + '_ins'
            if out_str in dd: 
                self._ann_to_spatial(dd, anns, type_str, out_str)
        return dd
#}}}
class DatasetTransform_NegativeTokensSample(DatasetTransform):#{{{
    def __init__(self, number=1, tokenids_key=['token_ids'], restore_key=['neg_tokens'],  runtime=False):
        """ not in runtime. time consuming
            
            must after : 
                DatasetTransform_Tokenize

            effect : 
                gather the $tokenids_key and sample number negative token samples and 
                store them in $restore_key
        """
        super(DatasetTransform_NegativeTokensSample, self).__init__(runtime)
        self._tokenids_key = tokenids_key
        self._restore_key = restore_key
        self._number = number

    def pre_process (self, dds, context):
        " ================= restore the pool information  ==================="
        all_tokens = []
        for dd in dds : 
            token_ids = self._get_by_search_key(dd, self._tokenids_key)
            assert type(token_ids) == list, "the token_ids must be list, but {} found!".format(type(token_ids))
            all_tokens.append(token_ids)
        self._all_tokens = all_tokens

    def process(self, dd, context):
        """ add spatial feature for every proposals

            need : dd['annotations']
            set  : dd['props_ins'].proposal_spatial = Tensor (N x 5)
        """
        n_len = len(self._all_tokens)
        indices = np.random.randint(0, n_len, self._number)
        neg_tokens = []
        for index in indices: 
            neg_tokens.append(copy.deepcopy(self._all_tokens[index]))
        self._set_by_search_key(dd, neg_tokens, self._restore_key)
        return dd
#}}}
class DatasetTransform_AddAttribute(DatasetTransform):#{{{
    def filter_low_frequent(self, number=5):
        keys = [ _[0] for _ in self.word2freq.items() if _[1] >= number ]
        keys = set(keys)
        tmp = [ _[0] for _ in self.word2id.items() if _[0] in keys ]
        print ('[DatasetTransform_AddAttribute] Sum atts:', sum([ self.word2freq[_] for _ in tmp ]))
        tmp = [ [_, i] for i, _ in enumerate(tmp) ]
        self.word2id = dict(tmp)
        print (self.word2id)
     
    def get_atts(self, atts):
        #print (atts)
        l = atts['r2'] + atts['r7']
        # r1 -- noun  r2 -- color  r7 -- location
        return [i for i in l if i != 'none']
    
    def __init__(self, filter_strategy=None, sent_json_filepath=None, padding_size=10, restore_atts_key=['atts'], restore_atts_mask=['atts_mask'],  runtime=False):
        """ not in runtime. time consuming  
            effect : 
                gather the $tokenids_key and sample number negative token samples and 
                store them in $restore_key
        """
        assert sent_json_filepath is not None, "Must Provide sent.json file"
        super(DatasetTransform_AddAttribute, self).__init__(runtime)
        with open(sent_json_filepath) as fp : 
            self.att_json = json.load(fp)
        self.padding_size = padding_size
        self._restore_atts_key = restore_atts_key
        self._restore_atts_mask = restore_atts_mask
        self.word2id = {}
        self.word2freq = {}
        self.sentid2atts = {}
        self.cnt = 0
        self.tot_atts = 0 # 所有的有效atts的个数，求平均值
        self._filter = filter_strategy
        for item in self.att_json:
            atts = item['atts']
            self.sentid2atts[item['sent_id']] = atts
            for tkn in self.get_atts(atts):
                self.tot_atts += 1
                self.word2freq[tkn] = self.word2freq.get(tkn, 0) + 1
                if tkn not in self.word2id:
                    self.word2id[tkn] = self.cnt
                    self.cnt = self.cnt + 1
        
        self.filter_low_frequent(10)
        num_atts = len(self.word2id)
        freq = np.zeros((num_atts,), 'int32')
        for key, idx in self.word2id.items():
            freq[idx] = self.word2freq[key]
        self.tns_freq = torch.as_tensor(freq).float()
        
        print ("[DatasetTransform_AddAttribute] Attribute Count: ", len(self.word2id), '/', self.cnt)
        print ("[DatasetTransform_AddAttribute] Attribute Set  : ", self.word2id.keys())

    def pre_process (self, dds, context):
        ...
        
    def filter_color_noun(self, dd, context, atts):
        if len([_ for _ in dd['token_ids'] if _ != 7]) <= 3 and len([_ for _ in atts['r2'] if _ != 'none']) > 0 : 
            return dd
        return None
    
    def filter_loc_noun(self, dd, context, atts):
        if len([_ for _ in dd['token_ids'] if _ != 7]) <= 6 and len([_ for _ in atts['r4'] if _ != 'none']) > 0 : 
            return dd
        return None

    def filter_long_sentence(self, dd, context, atts):
        if len([_ for _ in dd['token_ids'] if _ != 7]) > 4 :
            #print (dd['raw'])
            return dd
        return None

    def filter_union(self, dd, context, atts):
        ret = self.filter_color_noun(dd, context, atts)
        if ret: return ret 
        ret = self.filter_loc_noun(dd, context, atts)
        if ret: return ret
        return None

    def filter_short(self, dd, context, atts):
        if 'short' in dd['raw'].lower() :
            print (dd['raw'])
            return dd
        return None
    
    def filter_all(self, dd, context, atts):
        return dd
    
    def process(self, dd, context):
        """ 
        """
        num_atts = len(self.word2id)
        sent_id = self._get_by_search_key(dd, ['sent_id'])
        atts = self.sentid2atts[sent_id]
        ret = np.zeros((num_atts,), 'int32')
        for idx, tkn in enumerate(self.get_atts(atts)):
            if tkn not in self.word2id: continue
            tid = self.word2id[tkn]
            ret[tid] = 1
        self._set_by_search_key(dd, torch.as_tensor(ret).long() , self._restore_atts_key )
        self._set_by_search_key(dd, self.tns_freq, ['atts_freq'])
        fil = self.filter_all
        if self._filter == 'color': 
            fil = self.filter_color_noun
        if self._filter == 'location':
            fil = self.filter_loc_noun
        if self._filter == 'union':
            fil = self.filter_union
        if self._filter == 'long':
            fil = self.filter_long_sentence
        if self._filter == 'short':
            fil = self.filter_short
        return fil(dd, context, atts)
#}}}
class DatasetTransform_AddFrcnProposals(DatasetTransform):#{{{
    def __init__(self, loader, runtime=True):
        """ in runtime mode, because this should varies with iteration
        """
        super(DatasetTransform_AddFrcnProposals,self).__init__(runtime)
        self._loader = loader

    def process(self, dd, context):
        """ add spatial feature for every proposals

            set  : dd['props_ins']
                   dd['gt_ins']
                   dd['annotations']
        """
        def set_annoatations():#{{{
            filename = dd['file_name']
            sent_id = dd['sent_id']
            dd['annotations'] = self._loader.get(filename, sent_id)#}}}
        def set_task_related():#{{{
            """ 设置一些 task related 信息, 一些简化操作之类的， 放到这个更加高效，
                所有的都放到 ： batched_inputs 更目录下
                dd['gt_ins'] 和 dd['props_ins'] 是相同类型的Instance，可以concat。
                他们所有的属性都是一样的。
            """
            gt = dd['annotations'][0]
            dd['gt_ins'] = Instances((224,224), 
                                               boxes=Boxes([gt['bbox']]), 
                                               classes=torch.IntTensor(gt['class']),
                                               features=torch.from_numpy(np.array([gt['feat']])),
                                              )
            props = [ item['bbox'] for item in dd['annotations'][1:] ]
            assert (len(props) > 0)
            feats = [ item['feat'] for item in dd['annotations'][1:] ]
            classes = [ item['class'] for item in dd['annotations'][1:] ]
            dd['props_ins'] = Instances((224,224), 
                                        boxes=Boxes(props), 
                                        features=torch.from_numpy(np.array(feats)), 
                                        classes=torch.from_numpy(np.array(classes)))#}}}
        def only_apply_box(transformer, annotations):#{{{
            for annotation in annotations:
                if "bbox" in annotation:
                    bbox = annotation['bbox']
                    annotation["bbox"] = transformer.apply_box(bbox)[0]
                    annotation["bbox_mode"] = BoxMode.XYXY_ABS#}}}
        set_annoatations()
        only_apply_box(ScaleTransform(dd['height'], dd['width'], 224, 224), dd['annotations'])
        set_task_related()
        return dd
#}}}
class DatasetTransform_AddEasyProposals(DatasetTransform):#{{{
    def __init__(self, loader, neg_number, sample_ratio, color_file_name=None, runtime=True):#{{{
        """ in runtime mode, because this should varies with iteration
        """
        super(DatasetTransform_AddEasyProposals,self).__init__(runtime)
        self._loader = loader#}}}
        self._neg_number = neg_number
        self._sample_ratio = sample_ratio
        self._color_file_name = color_file_name
    def _ref_list_to_anns(self, ref_list, ttype): #{{{
        anns = []
        for ref in ref_list:
            sent_id = random.choice(ref['sent_ids'])
            negfeat, negbox = self._loader.get_gtfeat(sent_id)
            ann = {'type': ttype}
            ann['bbox_mode'] = BoxMode.XYXY_ABS
            ann['bbox'] = negbox # [[]]
            ann['class'] = 0
            ann['origin_bbox'] = copy.deepcopy(negbox)
            ann['feat'] = negfeat
            ann['sent_id'] = sent_id
            ann['raw'] = ref['sentences'][0]['raw']
            anns.append(ann)
        return anns#}}}
    def _negsample_proposals(self, context, image_id, sent_id, number=1, sample_ratio=0.5):#{{{
        REFER = context['REFER']
        ref_list = REFER.imgToRefs[image_id]
        ref_list_neg = [ _ for _ in ref_list if sent_id not in _['sent_ids'] ]
        ref_list_selected = []
        t_number = number
        while (t_number):
            t_number -= 1
            if len(ref_list_neg) and np.random.uniform(0, 1, 1) < sample_ratio:
                # sample negative in the same image
                neg = ref_list_neg[random.choice(range(len(ref_list_neg)))]
                ref_list_selected.append(neg)
            else:
                # sample negative in the global
                ref_list_selected.extend(self._global_referid_sample(context, image_id, sent_id, 1))
        assert ref_list_selected.__len__() == number, "expected {}, found {}".format(number, ref_list_selected.__len__())
        return self._ref_list_to_anns(ref_list_selected, 'neg')#}}}
    def _all_proposals_annotation(self, context, image_id, sent_id):#{{{
        REFER = context['REFER']
        ref_list = REFER.imgToRefs[image_id]
        ref_list = [ _ for _ in ref_list if sent_id not in _['sent_ids'] ]
        #assert (len(ref_list) > 0)
        return self._ref_list_to_anns(ref_list, 'prop')#}}}
    def _global_referid_sample(self, context, image_id, sent_id, number=1):#{{{
        REFER = context['REFER']
        ref_ids = context['ref_ids']
        ref_list = REFER.loadRefs([ random.choice(ref_ids) for _ in range(number) ])
        return ref_list #}}}
    def _gt_annotation(self, context, image_id, sent_id):#{{{
        REFER = context['REFER']
        ref_list = [REFER.sentToRef[sent_id]]
        assert (len(ref_list) > 0)
        return self._ref_list_to_anns(ref_list, 'gt')#}}}
    @staticmethod
    def calulate_color_bincount(image, mask=None, size=(10,10,10), ctype='rgb'):#{{{
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
#}}}
    @staticmethod
    def ann2cw(REFER, ann, file_name):#{{{
        """ for ann 2 color word / image color feature
        """
        cw_file = osp.join(color_word_dir, file_name)
        assert osp.exists(cw_file), "Error : Not Found color_word feature file, please set cfg.DATASETS.COLOR=True and num_worker=0 to regenerate"
        ref = REFER.sentToRef[ann['sent_id']]
        f = h5py.File(cw_file, "r")
        ref_name = ref['file_name']
        cw = np.array(f[ref_name])
        return cw
        #}}}
    def process(self, dd, context):#{{{
        """ add spatial feature for every proposals
            @set  : dd['props_ins']
                   dd['gt_ins']
                   dd['annotations']
        """
        filename = dd['file_name']
        sent_id = dd['sent_id']
        image_id = dd['image_id']
        gt_anns = self._gt_annotation(context, image_id, sent_id)
        neg_anns = self._negsample_proposals(context, image_id, sent_id, self._neg_number, self._sample_ratio)
        prop_anns = self._all_proposals_annotation(context, image_id, sent_id)
        REFER = context['REFER']

        def ann2instances(anns):#{{{
            """ 将 anns 转化为 instances
            """
            return Instances((224,224), 
                boxes=Boxes([ _['bbox'] for _ in anns ]), 
                classes=torch.IntTensor([ _['class'] for _ in anns ]),
                features=torch.from_numpy(np.array([ _['feat'] for _ in anns])).float(),
                color=np.array([DatasetTransform_AddEasyProposals.ann2cw(REFER, _, self._color_file_name) for _ in anns])
            ) #}}}
        def only_apply_box(transformer, annotations):#{{{
            for annotation in annotations:
                if "bbox" in annotation:
                    bbox = annotation['bbox']
                    annotation["bbox"] = transformer.apply_box(bbox)[0]
                    annotation["bbox_mode"] = BoxMode.XYXY_ABS#}}}

        only_apply_box(ScaleTransform(dd['height'], dd['width'], 224, 224), gt_anns)
        only_apply_box(ScaleTransform(dd['height'], dd['width'], 224, 224), prop_anns)
        only_apply_box(ScaleTransform(dd['height'], dd['width'], 224, 224), neg_anns)
        dd['gt_ins'] =    ann2instances(gt_anns)
        dd['props_ins'] = ann2instances(prop_anns)
        dd['negs_ins'] =  ann2instances(neg_anns)
        #print (gt_anns)
        assert (len(gt_anns) == 1)
        dd['annotations'] = gt_anns + prop_anns + neg_anns # g
        return dd#}}}
#}}}
class DatasetTransform_AddLocRelFeature(DatasetTransform):#{{{
    def __init__(self, proposal_loader=None, max_number=5, is_train=True, runtime=True):#{{{
        """ must in runtime mode, because in this time , we have annotations
            add the location features and relation features in the given 'gt', 'prop', 'neg' type annotations
        """
        super(DatasetTransform_AddLocRelFeature ,self).__init__(runtime)
        self._max_number = max_number
        self._loader = proposal_loader
        self._is_train = is_train
        #}}}
    @staticmethod
    def fetch_neighbors(ref_id, REFER):#{{{
        """ given ref_id, return the neighbors ref_ids
        """
        cur_ref = REFER.Refs[ref_id]
        ref_list = REFER.imgToRefs[cur_ref['image_id']]
        
        same_ref_ids = [ _['ref_id'] for _ in ref_list if ref_id != _['ref_id'] and cur_ref['category_id'] == _['category_id']]
        diff_ref_ids = [ _['ref_id'] for _ in ref_list if ref_id != _['ref_id'] and cur_ref['category_id'] != _['category_id']]
        def compare(ref_id0):
            ann_id0, ann_id1 = REFER.Refs[ref_id0]['ann_id'], cur_ref['ann_id']
            x, y, w, h = REFER.Anns[ann_id0]['bbox']
            ax0, ay0 = x+w/2, y+h/2
            x, y, w, h = REFER.Anns[ann_id1]['bbox']
            rx, ry = x+w/2, y+h/2
            # closer --> former
            return (rx-ax0)**2 + (ry-ay0)**2

        return sorted(same_ref_ids, key=lambda x : compare(x)), sorted(diff_ref_ids, key=lambda x: compare(x))
        #}}}
    @staticmethod
    def _cal_diff_spatial(box_i, box_j):#{{{
        """ box_i : (1 , 4)
            box_j : (n , 4)
            return: (5*max_number, ) 计算box_i为核心, box_j作为围绕的diff_spatials. 返回n*5
        """
        _max_number = 5
        output = np.zeros((_max_number * 5, ))
        if (box_j.shape[0] == 0): return output
        box_j = box_j[:_max_number,:]  # 截取

        width = box_i[0,2] - box_i[0,0]
        height = box_i[0,3] - box_i[0,1]
        ori_props = box_j - box_i
        ori_props[:,0] = ori_props[:,0] / width 
        ori_props[:,2] = ori_props[:,2] / width 
        ori_props[:,1] = ori_props[:,1] / height 
        ori_props[:,3] = ori_props[:,3] / height #  N x 4
        area_props     = (ori_props[:,2] - ori_props[:,0]) * (ori_props[:,3] - ori_props[:,1])
        area_props     = np.reshape(area_props, [-1,1])
        diff_spatial  = np.concatenate([ori_props, area_props], axis=1).reshape([-1])
        output[:len(diff_spatial)] = diff_spatial
        return output
        #}}}
    @staticmethod
    def _anns2refids(anns, REFER):#{{{
        sentids = [ ann['sent_id'] for ann in anns ]
        return [ REFER.sentToRef[sent]['ref_id'] for sent in sentids ]#}}}
    @staticmethod
    def _gather_boxes_from_refids(ref_ids, REFER):#{{{
        ann_ids =  [ REFER.Refs[i]['ann_id'] for i in ref_ids ]
        boxes = np.array([REFER.Anns[i]['bbox'] for i in ann_ids])
        if (boxes.shape[0]):
            boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        return boxes
     #}}}
    @staticmethod
    def _cal_loc_from_single_ann(ann, REFER, loader):#{{{
        """ all id means ref id
        """
        cur_id = DatasetTransform_AddLocRelFeature._anns2refids([ann], REFER)[0]
        same_ids, diff_ids = DatasetTransform_AddLocRelFeature.fetch_neighbors(cur_id, REFER)
        box_i = DatasetTransform_AddLocRelFeature._gather_boxes_from_refids([cur_id], REFER)
        same_boxes = DatasetTransform_AddLocRelFeature._gather_boxes_from_refids(same_ids, REFER)
        diff_boxes = DatasetTransform_AddLocRelFeature._gather_boxes_from_refids(diff_ids, REFER) # n, 5*max_number
        sent_ids =  [ REFER.Refs[ref]['sent_ids'][0] for ref in diff_ids ]
        cxt_vis = np.zeros((5, 256))
        if len(sent_ids) > 0:
            vis_feats = torch.Tensor([ loader.get_gtfeat(sid)[0] for sid in sent_ids ]).mean([2,3]).numpy()
            vis_feats = vis_feats[:5]
            cxt_vis[:vis_feats.shape[0]] = vis_feats
        return (DatasetTransform_AddLocRelFeature._cal_diff_spatial(box_i, same_boxes), 
                DatasetTransform_AddLocRelFeature._cal_diff_spatial(box_i, diff_boxes),
                cxt_vis)
        #}}}
    @staticmethod
    def _cal_cxt_vis_from_single_ann(ann, REFER, loader):#{{{
        cur_id = DatasetTransform_AddLocRelFeature._anns2refids([ann], REFER)[0]
        sent_ids =  ann['sent_id'] 
        return vis_feats
        #}}}
    def _ann_add_loc_rel_features(self, dd, context, anns: list, ann_type: str, ins_key: str):#{{{
        """ collect the ann_type as list and calculate spatials, then attach to dd['<ins_key>'] Instances.relation and Instances.location
        """
        REFER = context['REFER']
        dim_per_ref = self._max_number * 5
        height, width = dd['image_size']
        anns = [ ann for ann in anns if ann['type'] == ann_type ]
        if (len(anns) == 0): 
            dd[ins_key].same_rel_loc = torch.Tensor([]).reshape([0, dim_per_ref]).float()
            dd[ins_key].diff_rel_loc = torch.Tensor([]).reshape([0, dim_per_ref]).float()
            dd[ins_key].cxt_vis = torch.Tensor([]).reshape([0, 5, 256]).float()
            return 
        loc_features = [ self._cal_loc_from_single_ann(ann, REFER, self._loader) for ann in anns ]
        # print(loc_features[0][0].shape)
        same_loc = [ _[0] for _ in loc_features ]
        diff_loc = [ _[1] for _ in loc_features ]
        cxt_vis_list = [ _[2] for _ in loc_features ]
        dd[ins_key].same_rel_loc = torch.Tensor(same_loc)
        dd[ins_key].diff_rel_loc = torch.Tensor(diff_loc)
        dd[ins_key].cxt_vis = torch.Tensor(cxt_vis_list)
        #}}}
    def process_train(self, dd, context):#{{{
        """ 
        """
        anns = dd['annotations']
        type_set = set([ ann['type'] for ann in dd['annotations'] ])
        for type_str in type_set:
            out_str = type_str + 's_ins' if type_str != 'gt' else type_str + '_ins'
            self._ann_add_loc_rel_features(dd, context, anns, type_str, out_str)
        return dd
        #}}}
    def create_fake(self, anns) : #{{{
        class fake_loader:
            def __init__(self, anns):
                self.anns = anns
            def get_gtfeat(self, sid):
                return [self.anns[sid]['feat']]

        class fake_refer: 
            def __init__(self):
                self.Refs = {}
                self.Anns = {}
                self.sentToRef = {}
                self.imgToRefs = None

        res = fake_refer()
        res_loader = fake_loader(anns)
        anns = [ _ for _ in anns if _['type'] == 'prop' ]
        for idx, ann in enumerate(anns):
            ann['sent_id'] = idx
            res.Refs[idx] = {'ann_id': idx, 'ref_id': idx, 'category_id': ann['class'], 'image_id': 0, 'sent_ids':[idx]}
            res.Anns[idx] = {'bbox': list(ann['bbox'])}
            res.sentToRef[idx] = res.Refs[idx]
        res.imgToRefs = {0: list(res.Refs.values())}
        return res, res_loader
#}}}
    def process_eval(self, dd, context):#{{{
        """ 
        """
        anns = dd['annotations']
        gt = [ _ for _ in anns if _['type'] == 'gt' ][0]
        gt['sent_id'] = dd['sent_id']  # 给 gt 添加 sent_id
        self._ann_add_loc_rel_features(dd, context, anns, 'gt', 'gt_ins')
        fake_refer, fake_loader = self.create_fake(anns)
        fake_context = {}
        fake_context['REFER'] = fake_refer
        old_loader = self._loader
        self._loader = fake_loader
        self._ann_add_loc_rel_features(dd, fake_context, anns, 'prop', 'props_ins')
        self._loader = old_loader
        del fake_refer
        del fake_loader
        return dd
        #}}}
    def process(self, dd, context):#{{{
        if self._is_train: 
            return self.process_train(dd, context)
        else : 
            return self.process_eval(dd, context)
    #}}}
#}}}
def process_cw_func(args):#{{{
    image_path, ref_name, box = args
    image = read_image(osp.join(image_dir, image_path), format='RGB')
    image = image.transpose((2,0,1))
    mask  = np.zeros_like(image)
    mask[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 255
    cw = DatasetTransform_AddEasyProposals.calulate_color_bincount(image, mask, size=(10,10,10), ctype='rgb')
    cw = cw / (1.0 * cw.sum())
    assert (cw.shape == (1000,))
    return ref_name, cw#}}}
class ReferitFastDataset(BaseDataset):
    """ 这个数据集可以使用四个类型的子数据#{{{
        referit_refcocog_<split>_train|val|test
        referit_refclef_<split>_train|val|test
        referit_refcoco_<split>_train|val|test
        referit_refcoco+_<split>_train|val|test

        需要的Cfg：
            DATASETS.SPLIT = ["unc", "goo", ]
    """#}}}
    def _set_proposal_root(self, cfg, is_train):#{{{
        self._proposal_root = cfg.DATASETS.get("PHRASE_GROUNDING", None)
        if self._proposal_root: 
            self._proposal_root = cfg.DATASETS.PHRASE_GROUNDING.TRAIN_PROPOSAL_ROOT if is_train else cfg.DATASETS.PHRASE_GROUNDING.TEST_PROPOSAL_ROOT#}}}
    def _lazy_init(self):#{{{
        """ 
            the time cost or memory cost initialization is located here.

            h5py is created here, because the h5py is not 
        """
        self.need_lazy_init = False
        self.proposal_dict = None
        print ("Start Lazy Initialized!!")
        if self._proposal_root:
            with open(osp.join(self._proposal_root, 'proposals.pkl'), 'rb') as fp : 
                self.prop_loader = ProposalLoader(self._proposal_root)
                self.proposal_dict = self.prop_loader.proposal_dict

        self._dataset_transforms = self._get_static_dataset_transforms() + self._get_runtime_dataset_transforms()
        self.dataset_dicts, self.context = self._start_dataset_transform(self.dataset_dicts,
            self._dataset_transforms, 
            {
                'proposal_dict': self.proposal_dict, 
                'REFER': self.REFER,
                'ref_ids': self.ref_ids, 
            }
        )
        self.need_lazy_init = False
#}}}
    def process_cw(self):#{{{
        self.color_file_name = color_file_name = "color_cw_{}_{}.hdf5".format(self.dataset_name, self.dataset_split)
        if not osp.exists(osp.join(color_word_dir, color_file_name)):
            print("Start Caching Color Word Feature")
            os.system("mkdir -p " + color_word_dir)

            f = h5py.File(osp.join(color_word_dir, self.color_file_name), "w")
            file_name_sets = set()
            args = []
            for ref in tqdm.tqdm(self.REFER.loadRefs(self.ref_ids)):
                image_path = self.REFER.loadImgs(ref['image_id'])[0]['file_name'] 
                ref_name = ref['file_name']
                if ref_name in file_name_sets: continue
                file_name_sets.add(ref_name)
                boxes = np.array([self.REFER.Anns[ref['ann_id']]['bbox']])
                box = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)[0]
                args.append((image_path, ref_name, box))

            core_num = 12
            import multiprocessing
            with multiprocessing.Pool(processes=core_num) as pool : 
                results = list(tqdm.tqdm(pool.imap(process_cw_func, args), total=len(args)))

            import pdb
            pdb.set_trace()
            for res in results:
                ref_name, cw = res
                f.create_dataset(ref_name, data=cw)
            f.close()#}}}
    def __init__(self, cfg, raw_name, transforms=[], is_train=True):#{{{
        super(ReferitFastDataset, self).__init__(cfg, raw_name, transforms, is_train)
        self.cfg = cfg
        self.need_lazy_init = True
        self.refer_tool_root = cfg.DATASETS.get("ROOT", "/home/data/dataset/cv/referit/refer/")
        self._output_dir = cfg.OUTPUT_DIR
        self._verbose = cfg.DATASETS.get("VERBOSE", False)
        self._eval_type = cfg.DATASETS.get("EVAL_TYPE", "phrasegrounding")
        self._unique = cfg.DATASETS.get("UNIQUE", False)
        self._is_train = is_train
        self.raw_name = raw_name
        self._set_proposal_root(cfg, is_train)
        self.dataset_name = self.raw_name.split("_")[1]
        self.dataset_splitby = self.raw_name.split("_")[2]
        self.dataset_split = self.raw_name.split("_")[3]
        self._add_attribute = True
        self._test_proposal_type = 'gt' if is_train else cfg.DATASETS.get("TEST_TYPE", "gt")
        self._filter = cfg.DATASETS.get('FILTER', None)
        self._color_histo = cfg.DATASETS.get('COLOR', False)

        assert self._test_proposal_type in ['det', 'gt'], "cfg.DATASETS.TEST_TYPE must in ['det', 'gt']"

        if self.dataset_name == "refclef":
            self.image_root = osp.join(self.refer_tool_root, 'data/images/saiapr_tc-12/')
        if self.dataset_name == "refcoco":
            self.image_root = osp.join(self.refer_tool_root, 'data/images/mscoco/train2014'.format(self.dataset_split))
        else : 
            assert (False)

        self._load_annotations()
        old_dataset_dicts = copy.deepcopy(self.dataset_dicts)
        self.proposal_dict = None
        if self._proposal_root:
            self.prop_loader = ProposalLoader(self._proposal_root)
            self.proposal_dict = self.prop_loader.proposal_dict
            self.prop_loader.close()
            del self.prop_loader

        self.dataset_dicts, _ = self._start_dataset_transform(self.dataset_dicts, self._get_static_dataset_transforms(), 
            {'proposal_dict': self.proposal_dict}
        )
        self.dsize = len(self.dataset_dicts)
        self.dataset_dicts = old_dataset_dicts
        if self._color_histo:
            self.process_cw()
        self._set_group()
        self.meta = self._get_metadata() #}}}
    def _get_static_dataset_transforms(self):#{{{
        cfg = self.cfg
        _dataset_transforms = []
        _dataset_transforms.extend([   # Add non runtime transformer
            DatasetTransform_Unknowlize_Truncate(False, -1, 15, ['tokens'], ['token_ids']), 
            DatasetTransform_Tokenize(False, True, ['token_ids'], ['token_ids'], self._is_train, osp.join(self._output_dir, "vocabulary.pkl")),
            DatasetTransform_ExcludeEmptyProposal(),
        ])

        if self._add_attribute : 
            atts_json_file = osp.join(self.refer_tool_root, "data/", "parsed_atts", self.dataset_name+'_'+self.dataset_splitby, "sents.json")
            _dataset_transforms.append(DatasetTransform_AddAttribute(self._filter, sent_json_filepath=atts_json_file))
            ...
        _dataset_transforms = [ _ for _ in _dataset_transforms if _ is not None ]
        return _dataset_transforms #}}}
    def _get_runtime_dataset_transforms(self): #{{{
        # Add runtime transformer
        _dataset_transforms = []
        _dataset_transforms.extend([   
            DatasetTransform_AddFrcnProposals(self.prop_loader, runtime=True) if self._test_proposal_type == 'det' else None,
            DatasetTransform_AddEasyProposals(self.prop_loader,
                self.cfg.DATASETS.NUM_NEG_REGION, self.cfg.DATASETS.SAMPLE_RATIO, self.color_file_name, 
                runtime=True) if self._test_proposal_type == 'gt' else None,
            DatasetTransform_AddSpatialFeature(runtime=True),  # Must after AddFrcnProposals
            DatasetTransform_AddLocRelFeature(self.prop_loader, 5, self._is_train, runtime=True),
            DatasetTransform_NegativeTokensSample(1, ['token_ids'], ['neg_token_ids'], runtime=True), 
        ])
        """ for conveninent, we add some None in the transformer, so we filter it out here
        """
        _dataset_transforms = [ _ for _ in _dataset_transforms if _ is not None ]
        return _dataset_transforms
        #}}}
    @staticmethod
    def _collect_for_mattnet(dd, keys):#{{{
        ret = {}
        for key in keys:
            if isinstance(dd[key], Instances): 
                ret[key] = dd[key].dataloader_speedup()
            else : 
                ret[key] = dd[key]
        return ret#}}}
    def __getitem__(self, idx):#{{{
        """
            output : {
            ## the following is the need of cvpack2: 
                'annotations' : 
                [ # 这些boxes是针对增广后的boxes
                {
                    'type': 'gt' | 'prop', 
                    'bbox': np.array([a, b, c, d]), 
                    'origin_bbox': np.array([a,b,c,d]), 
                    'bbox_mode':  BoxMode.ENUM, 
                    'feat' : np.array(Nx256x7x7) , 
                }
                ], 

            ## the following is the original information stored in referit dataset
                'file_name': image_file_str, 
                'tokens': [word1, word2, word3 ...],
                'image_size': [h, w] # image_size 是增广之后的 image 大小
                'height': h # 真实的image height
                'width' : w # 真实的image width
                'raw': str, 
                'bbox': [x, y, w, h]
                'category_id': cid,
                
            ## the following is preprocessed by dataset. contained image load + token2id
                'token_ids': [id1, id2, id3 ...],
                'neg_token_ids': [id1, id2, id3 ...],
            }
        """
        if self.need_lazy_init: 
            self._lazy_init()

        # ======== start runtime transform
        dd = dataset_dict = deepcopy(self.dataset_dicts[idx])
        dataset_dict = self._runtime_dataset_transform(dataset_dict, self._dataset_transforms)
        dataset_dict['atts'] = dd['atts'].numpy()
        dataset_dict['atts_freq'] = dd['atts_freq'].numpy()
        if self._is_train: # make the training process faster
            dataset_dict = self._collect_for_mattnet(dataset_dict, ['file_name', 'gt_ins', 'props_ins', 'negs_ins', 'token_ids', 'atts', 'atts_freq', 'neg_token_ids', 'raw'])

        return dataset_dict#}}}#}}}
    def __len__(self):#{{{
        #return len(self.dataset_dicts)
        return self.dsize
        #}}}
    def _set_group(self):#{{{
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)#}}}
    def _get_metadata(self):#{{{
        meta = {
            "evaluator_type": self._eval_type
        }
        return meta#}}}
    def _load_annotations(self):#{{{
        """ referit dataset have many information: 
                sentences / tokens / raw 
                annId -> bbox + mask infor
                imgId -> file name + :
        """
        timer = Timer()
        _REFER = refer.REFER
        self.REFER = _REFER(data_root=osp.join(self.refer_tool_root, "data/"), dataset=self.dataset_name, splitBy=self.dataset_splitby)

        dataset_dicts = []
        split = self.dataset_split
        #print (self.REFER.sentToRef['620317'])
        ref_ids = self.REFER.getRefIds(split=split)
        self.ref_ids = ref_ids
                
        for ref_id in tqdm.tqdm(ref_ids):
            ref_dict = self.REFER.loadRefs(ref_id) [0]
            ann_dict = self.REFER.loadAnns(ref_dict['ann_id'])[0]
            img_dict = self.REFER.loadImgs(ref_dict['image_id'])[0]
            cat_dict = self.REFER.loadCats(ref_dict['category_id'])[0]
            
            sentences = ref_dict['sentences'] 
            sentences = sentences[0:1] if self._unique else sentences
            for sentence in sentences:
                dd = {}
                dd['image_id'] = ref_dict['image_id']
                dd['sent_id'] = sentence['sent_id']
                if self._verbose:
                    dd['img_dict'] = img_dict
                    dd['ref_dict'] = ref_dict
                    dd['cat_dict'] = cat_dict
                    dd['ann_dict'] = ann_dict

                dd['file_name'] = img_dict['file_name']
                dd['image_size'] = [img_dict['height'], img_dict['width']]
                dd['tokens'] = sentence['tokens']
                dd['raw'] = sentence['raw']
                dd['category_id'] = None # cat_dict[] # FIXME (增加cat的逻辑)
                dd['bbox'] = self.REFER.getRefBox(ref_id)
                if len(dd['bbox']) != 4: 
                    print('invalid bbox')
                    continue ; 
                dd['height'] = img_dict['height']
                dd['width'] = img_dict['width']
                # FIXME (增加gt_classes信息)
                b = dd['bbox']
                dataset_dicts.append(dd)
        
        self.dataset_dicts = dataset_dicts
        logging.info("Loading {} takes {:.2f} seconds.".format(self.raw_name, timer.seconds()))
        #}}}
    def _start_dataset_transform(self, dataset_dicts, dataset_tranforms, pre_context_dict=None) : #{{{
        """ 总体逻辑，对于所有的transform依次执行：
                pre_process 所有的数据
                process 单条数据，并且进行filter
                然后将得到的dds输入到下一个管道中
                if runtime == True : means __getitem__() function, can't return None
                if runtime == False: means _load_annotations() function
        """
        timer = Timer()
        context = DatasetTransformContext()
        if pre_context_dict : 
            for k, v in pre_context_dict.items():
                context[k] = v
                
        for transform in dataset_tranforms : 
            #print (dataset_dicts[0])
            transform.pre_process(dataset_dicts, context)
            new_dataset_dicts = []
            for dd in dataset_dicts:
                if dd and transform._runtime == False : 
                    dd = transform.process(dd, context)
                if dd : new_dataset_dicts.append(dd)
            dataset_dicts = new_dataset_dicts

        logging.info("DatasetTransform {} takes {:.2f} seconds.".format(self.raw_name, timer.seconds()))
        return dataset_dicts, context
#}}}
    def _runtime_dataset_transform(self, dd, dataset_tranforms) : #{{{
        """ runtime stage transform proceduce
        """
        timer = Timer()
        context = self.context # reuse the global information
        assert dd, "input dd is None, Some wrong happen in your dataset or you put None object in self.dataset_dict"
        for transform in dataset_tranforms : 
            if transform._runtime == True:
                dd = transform.process(dd, context)
                assert dd, "return value can't be None, when in runtime mode"
        #logging.info("Runtime DatasetTrasnform {} takes {:.2f} seconds.".format(self.raw_name, timer.seconds()))
        self.context = context # reset
        return dd#}}}

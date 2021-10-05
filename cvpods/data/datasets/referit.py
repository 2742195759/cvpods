import tqdm
import json
import logging
import copy
import os
import os.path as osp

import numpy as np
import torch

#from brainpp.oss import OSSPath  删除
from copy import deepcopy
from cvpods.utils import Timer
import cvpods

from ..base_dataset import BaseDataset
from ..detection_utils import annotations_to_instances, filter_empty_instances
from .paths_route import _PREDEFINED_SPLITS_REFERIT
from cvpods.structures import Boxes, Instances, BoxMode
from cvpods.data.transforms import IoUCropTransform, ScaleTransform
from ..detection_utils import (annotations_to_instances, check_image_size,
                               create_keypoint_hflip_indices,
                               filter_empty_instances, read_image)
import os
import os.path as osp
import pickle

# FIXME : bugs here, if this code put in __init__, then multi-gpu will get : refer module not found
refer_tool_root = "/home/data/dataset/cv/referit/refer/"
refer = cvpods.utils.imports.dynamic_import('refer', refer_tool_root)

"""
special tokens : 
<eos>  <sos>  unknow
"""

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
        assert key in self._context_dict, "The key your tranform need is not in global context"
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
            if self._voc_freq[token] < filter_freq : new_tokens.append('unknow')
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
            word2id, id2word = pickle.load(open(self._save_file, 'rb')) 
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
            pickle.dump([word2id, id2word], open(self._save_file, 'wb'))

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
        file_name = dd['file_name']
        if file_name not in context['proposal_dict'] or len(context['proposal_dict'][file_name]['proposals']) == 0: 
            return None 
        else :
            return dd   #}}}
    
class DatasetTransform_AddSpatialFeature(DatasetTransform):#{{{
    def __init__(self, runtime=True):
        """ must in runtime mode, because in this time , we have annotations
        """
        super(DatasetTransform_AddSpatialFeature,self).__init__(runtime)

    def process(self, dd, context):
        """ add spatial feature for every proposals

            need : dd['annotations']
            set  : dd['props_ins'].proposal_spatial = Tensor (N x 5)
        """
        height, width = dd['image_size']
        image_area = height * width
        anns = dd['annotations']
        props_ins = dd['props_ins']
        box_list = [ ann['origin_bbox'] for ann in anns if ann['type'] == 'prop' ]
        ori_props = np.array(box_list)
        ori_props[:,0] = ori_props[:,0] / width 
        ori_props[:,2] = ori_props[:,2] / width 
        ori_props[:,1] = ori_props[:,1] / height 
        ori_props[:,3] = ori_props[:,3] / height #  N x 4
        area_props     = (ori_props[:,2] - ori_props[:,0]) * (ori_props[:,3] - ori_props[:,1])
        area_props     = np.reshape(area_props, [-1,1])
        spatial_feats  = np.concatenate([ori_props, area_props], axis=1)
        props_ins.proposal_spatial = torch.Tensor(spatial_feats)
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
        self._tokenids_key = tokenids_key
        self._restore_key = restore_key
        self._number = number
        super(DatasetTransform_NegativeTokensSample, self).__init__(runtime)

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
        l = atts['r2'] + atts['r7']
        return [i for i in l if i != 'none']
    
    def __init__(self, sent_json_filepath=None, padding_size=10, restore_atts_key=['atts'], restore_atts_mask=['atts_mask'],  runtime=False):
        """ not in runtime. time consuming  
            effect : 
                gather the $tokenids_key and sample number negative token samples and 
                store them in $restore_key
        """
        assert sent_json_filepath is not None, "Must Provide sent.json file"
        super(DatasetTransform_AddAttribute, self).__init__(runtime)
        self.att_json = json.load(open(sent_json_filepath))
        self.padding_size = padding_size
        self._restore_atts_key = restore_atts_key
        self._restore_atts_mask = restore_atts_mask
        self.word2id = {}
        self.word2freq = {}
        self.sentid2atts = {}
        self.cnt = 0
        self.tot_atts = 0 # 所有的有效atts的个数，求平均值
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
        print ("[DatasetTransform_AddAttribute] Attribute Count: ", len(self.word2id), '/', self.cnt)
        print ("[DatasetTransform_AddAttribute] Attribute Set  : ", self.word2id.keys())

    def pre_process (self, dds, context):
        ...
        
    def process(self, dd, context):
        """ 
        """
        sent_id = self._get_by_search_key(dd, ['sent_id'])
        atts = self.sentid2atts[sent_id]
        ret = np.zeros((self.padding_size,), 'uint32')
        mask = np.zeros((self.padding_size,), 'uint32')
        freq = np.zeros((self.padding_size,), 'uint32')
        for idx, tkn in enumerate(self.get_atts(atts)):
            if tkn not in self.word2id: continue
            tid = self.word2id[tkn]
            ret[idx] = tid
            mask[idx] = 1
            freq[idx] = self.word2freq[tkn]
        self._set_by_search_key(dd, ret,  self._restore_atts_key )
        self._set_by_search_key(dd, mask, self._restore_atts_mask)
        self._set_by_search_key(dd, mask, ['atts_freq'])
        return dd
#}}}

class ReferitDataset(BaseDataset):
    """ 这个数据集可以使用四个类型的子数据
        referit_refcocog_<split>_train|val|test
        referit_refclef_<split>_train|val|test
        referit_refcoco_<split>_train|val|test
        referit_refcoco+_<split>_train|val|test

        需要的Cfg：
            DATASETS.SPLIT = ["unc", "goo", ]
    """
    def _set_proposal_root(self, cfg, is_train):
        self._proposal_root = cfg.DATASETS.get("PHRASE_GROUNDING", None)
        if self._proposal_root: 
            self._proposal_root = cfg.DATASETS.PHRASE_GROUNDING.TRAIN_PROPOSAL_ROOT if is_train else cfg.DATASETS.PHRASE_GROUNDING.TEST_PROPOSAL_ROOT

    def __init__(self, cfg, raw_name, transforms=[], is_train=True):
        super(ReferitDataset, self).__init__(cfg, raw_name, transforms, is_train)
        
        self.refer_tool_root = cfg.DATASETS.get("ROOT", "/home/data/dataset/cv/referit/refer/")
        self._output_dir = cfg.OUTPUT_DIR
        self._verbose = cfg.DATASETS.get("VERBOSE", False)
        self._eval_type = cfg.DATASETS.get("EVAL_TYPE", "phrasegrounding")
        self._is_train = is_train
        self._image_only = cfg.DATASETS.get("IMAGE_ONLY", False) # 相同的 Image 将会只输入一次
        self.raw_name = raw_name
        self._set_proposal_root(cfg, is_train)
        self.dataset_name = self.raw_name.split("_")[1]
        self.dataset_splitby = self.raw_name.split("_")[2]
        self.dataset_split = self.raw_name.split("_")[3]
        self._add_attribute = True
        if self.dataset_name == "refclef":
            self.image_root = osp.join(self.refer_tool_root, 'data/images/saiapr_tc-12/')
        if self.dataset_name in ["refcoco", "refcocog"]:
            self.image_root = osp.join(self.refer_tool_root, 'data/images/mscoco/train2014'.format(self.dataset_split))

        self._REFER = refer.REFER
        self._load_annotations()
        self._set_group()
        self.meta = self._get_metadata()
        
    @staticmethod
    def _collect_for_mattnet(dd, keys):
        ret = {}
        for key in keys:
            ret[key] = dd[key]
        return ret

    def __getitem__(self, idx):
        """
            output : {
            ## the following is the need of cvpods: 
                'annotations' : 
                [ # 这些boxes是针对增广后的boxes
                {
                    'type': 'gt' | 'prop', 
                    'bbox': np.array([a, b, c, d]), 
                    'origin_bbox': np.array([a,b,c,d]), 
                    'bbox_mode':  BoxMode.ENUM, 
                    'feat' : np.array(Nx1024) , 
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
                'image': [image_tensor]
                'bbox_image': [bbox_image_tns1, bbox_image_tns2, ...],  和增广有关
                'token_ids': [id1, id2, id3 ...],
                'neg_token_ids': [id1, id2, id3 ...],

            ## the following will appear when DATASETS.VERBOSE = True, we can use it for debug, but will slow down our speed
                'img_dict': loadImgs()[0], 
                'ann_dict': loadAnns()[0], 
                'ref_dict': loadRefs()[0],
                'cat_dict': loadCats()[0],
            }
        """
        def process(img):
            """ 将图像格式转化为，NxCxHxW 或者 CxHxW
            """

            if len(img.shape) == 3:
                image_shape = img.shape[:2]  # h, w
                return torch.as_tensor(
                    np.ascontiguousarray(img.transpose(2, 0, 1)))
            elif len(img.shape) == 4:
                image_shape = img.shape[1:3]
                # NHWC -> NCHW
                return atorch.as_tensor(
                    np.ascontiguousarray(img.transpose(0, 3, 1, 2)))

        dataset_dict = deepcopy(self.dataset_dicts[idx])
        dd = dataset_dict
        image = read_image(osp.join(self.image_root, dataset_dict['file_name']), 'BGR')
#        if tuple(image.shape[0:2]) != (dd['height'], dd['width']):
#            image = image.transpose((1,0,2))
        assert tuple(image.shape[0:2]) == (dd['height'], dd['width']), "Refer dataset: Shape Error!!"

        dataset_dict['raw_image'] = image
        bbox = [int(i) for i in dataset_dict['bbox']]
        def set_annoatations():
            filename = dataset_dict['file_name']
            anns = []
            gt = {'type': 'gt'} 
            gtbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            gt.update({'bbox': gtbox, 'origin_bbox': gtbox.copy(), 'bbox_mode': BoxMode.XYXY_ABS})
            anns.append(gt)
            if self._proposal_root: 
                feat_file_name = osp.join(self._proposal_root, 'features', self.proposal_dict[filename]['feature']+'.npy')
                assert feat_file_name, "feat name error"
                #feats = self._reader.read(feat_file_name, to_numpy=True)
                feats = np.load(feat_file_name)
                if len(self.proposal_dict[filename]['proposals']) == 0: 
                    self.proposal_dict[filename]['proposals'].append([0, 0, 100, 100])
                    self.proposal_dict[filename]['class'].append([80])
                for idx, box in enumerate(self.proposal_dict[filename]['proposals']) : 
                    ann = {'type': 'prop'}
                    ann['bbox_mode'] = BoxMode.XYXY_ABS
                    ann['bbox'] = box
                    if 'classes' in self.proposal_dict[filename]:
                        ann['class'] = self.proposal_dict[filename]['classes'][idx]
                        
                    ann['origin_bbox'] = copy.deepcopy(box)
                    ann['feat'] = feats[idx]
                    anns.append(ann)
            dataset_dict['annotations'] = anns

        def set_task_related():
            """ 设置一些 task related 信息, 一些简化操作之类的， 放到这个更加高效，
                所有的都放到 ： batched_inputs 更目录下
            """
            gt = dd['annotations'][0]
            dataset_dict['gt_ins'] = Instances((224,224), gt_boxes=Boxes([gt['bbox'].tolist()]), gt_classes=torch.IntTensor([1]))
            props = [ item['bbox'] for item in dd['annotations'][1:] ]
            feats = [ item['feat'] for item in dd['annotations'][1:] ]
            classes = [ item['class'] for item in dd['annotations'][1:] ]
            dd['props_ins'] = Instances((224,224), proposal_boxes=Boxes(props), features=torch.from_numpy(np.array(feats)), classes=torch.from_numpy(np.array(classes)))
                
        set_annoatations()
        bbox_image, _ = IoUCropTransform(*bbox)(image, None)
        image, dataset_dict['annotations'] = ScaleTransform(* (dataset_dict['image_size']+[224,224]))(image, dataset_dict['annotations'])
        try : 
            bbox_image, _ = ScaleTransform(* (dataset_dict['bbox'][-1:-3:-1]+[224,224]))(bbox_image, None)
        except : 
            import pdb
            pdb.set_trace()
        
        set_task_related()
        dataset_dict['image'] = process(image)
        dataset_dict['bbox_image'] = process(bbox_image)
        # ======== start runtime transform
        if not self._image_only : 
            dataset_tranforms = [
                DatasetTransform_AddSpatialFeature(), 
            ]
            dataset_dict = self._runtime_dataset_transform(dataset_dict, dataset_tranforms)
        return dataset_dict
        
    def __len__(self):
        return len(self.dataset_dicts)

    def _set_group(self):
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)

    def _get_metadata(self):
        meta = {
            "evaluator_type": self._eval_type
        }
        return meta

    def _load_annotations(self):
        """ referit dataset have many information: 
                sentences / tokens / raw 
                annId -> bbox + mask infor
                imgId -> file name + :
        """
        timer = Timer()
        self.REFER = self._REFER(data_root=osp.join(self.refer_tool_root, "data/"), dataset=self.dataset_name, splitBy=self.dataset_splitby)

        dataset_dicts = []
        split = self.dataset_split
        ref_ids = self.REFER.getRefIds(split=split)
        self.proposal_dict = None
        if self._proposal_root:
            with open(osp.join(self._proposal_root, 'proposals.pkl'), 'rb') as fp : 
                self.proposal_dict = pickle.load(fp)
                
        for ref_id in tqdm.tqdm(ref_ids):
            ref_dict = self.REFER.loadRefs(ref_id) [0]
            ann_dict = self.REFER.loadAnns(ref_dict['ann_id'])[0]
            img_dict = self.REFER.loadImgs(ref_dict['image_id'])[0]
            cat_dict = self.REFER.loadCats(ref_dict['category_id'])[0]
            
            for sentence in ref_dict['sentences']:
                dd = {}
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
        
        logging.info("Loading {} takes {:.2f} seconds.".format(self.raw_name, timer.seconds()))
        dataset_tranforms = [
            #DatasetTransform_Unknowlize_Truncate(False, -1, 15, ['tokens'], ['token_ids']), 
            #DatasetTransform_Tokenize(False, True, ['token_ids'], ['token_ids'], self._is_train, osp.join(self._output_dir, "vocabulary.pkl")), 
            #DatasetTransform_NegativeTokensSample(1, ['token_ids'], ['neg_token_ids'])
        ]
        
        if not self._image_only : 
            dataset_tranforms.append(DatasetTransform_ExcludeEmptyProposal())
            

        #{{{ add this logic to judge : whether small gtboxes will cause no proper proposals
        self._exclude_small_gtboxes = not self._is_train and not self._image_only
        if self._exclude_small_gtboxes : 
            dataset_tranforms.append(DatasetTransform_ExcludeSmallGt(20000)),
        #}}}
        if self._add_attribute : 
            atts_json_file = osp.join(self.refer_tool_root, "data/", "parsed_atts", self.dataset_name+'_'+self.dataset_splitby, "sents.json")
            dataset_tranforms.append(DatasetTransform_AddAttribute(sent_json_filepath=atts_json_file))

        dataset_dicts, context = self._start_dataset_transform(dataset_dicts,  dataset_tranforms, {'proposal_dict': self.proposal_dict})
        self.context = context
        
        if not self._is_train and not self._image_only : #{{{
            #indices = np.random.randint(0, len(dataset_dicts), 1000)
            #self.dataset_dicts = [dataset_dicts[i] for i in indices]
            
            self.dataset_dicts = dataset_dicts
        else : 
            self.dataset_dicts = dataset_dicts#}}}

        # if image_only ： 那么将所有相同名字的过滤掉
        if self._image_only : 
            print('dataset_len:', len(dataset_dicts))
            print('Image Only Mode!!')
            image_pool = {}
            new_dict = []
            for item in self.dataset_dicts:
                if item['file_name'] not in image_pool:
                    new_dict.append(item)
                    image_pool[item['file_name']] = 1
            print ('dataset_len:', len(new_dict))
            #self.dataset_dicts = new_dict
            

    def _start_dataset_transform(self, dataset_dicts, dataset_tranforms, pre_context_dict=None) : 
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

    def _runtime_dataset_transform(self, dd, dataset_tranforms) : 
        """ runtime stage transform proceduce
        """
        timer = Timer()
        context = self.context # reuse the global information
        assert dd, "input dd is None, Some wrong happen in your dataset or you put None object in self.dataset_dict"
        for transform in dataset_tranforms : 
            if transform._runtime == True:
                dd = transform.process(dd, context)
                assert dd, "return value can't be None, when in runtime mode"
        logging.info("Runtime DatasetTrasnform {} takes {:.2f} seconds.".format(self.raw_name, timer.seconds()))
        self.context = context # reset
        return dd

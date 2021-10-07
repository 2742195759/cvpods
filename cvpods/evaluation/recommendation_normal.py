import copy
import itertools
import logging
import os
import copy
from collections import OrderedDict

import torch

from cvpods.utils import PathManager, comm, create_small_table

from .evaluator import DatasetEvaluator
from .registry import EVALUATOR


@EVALUATOR.register()
class RecommendationEvaluator(DatasetEvaluator):
    """
    Evaluate normal recommendation test dataset
    """
    def __init__(self, dataset_name, meta, cfg, distributed=True, dump=True):
        """ 
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            meta (SimpleNamespace): dataset metadata.
            cfg (CfgNode): cvpods Config instance.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
        """
        self._distributed = distributed
        self._logger = logging.getLogger(__name__)
        self._metadata = meta
        self._dump = dump

        self._results = []
        self._top_k = cfg.DATASETS.get("top_k", None)
        self._output_file = os.path.join(cfg.OUTPUT_DIR, "metric_" + 
            dataset_name.split("_")[1])

        if not self._top_k : self._top_k = [3,5,10,]
    
    def reset(self):
        self._results = []
    
    # @interface
    def process(self, inputs, outputs):
        """
        @param:
            inputs : [
                dict(
                    "user": int64
                    "gt"  : [ int64 ]
                    "candidate": [ int64 ]
                )
            ]
            outputs : 
                dict() {
                    "score": [float64]
                }

        """
        # print (type(outputs))
        # print (outputs)

        res = copy.deepcopy(inputs[0])
        assert isinstance(outputs, dict)
        assert isinstance(outputs["score"], list) and len(outputs["score"]) == len(inputs[0]["candidate"]) and isinstance(outputs["score"][0], float), "return in eval model is not satisfied prototype of RecommendationEvaluator: score -> [float64]"
        res["score"] = copy.deepcopy(outputs["score"])
        self._results.append(res)
       
    def evaluate(self):
        return_dict = OrderedDict()
        for k in self._top_k:
            return_dict.update(self.evaluate_k(k))

        if self._dump : 
            self.dump(return_dict)

        return return_dict

    def evaluate_k(self, k):
        ret = {}
        p = self.precise(k)
        r = self.recall(k)
        ret['HR@'+str(k)] = self.hitratio(k)
        ret['P@'+str(k)] = p
        ret['R@'+str(k)] = r
        ret['F1@'+str(k)] = 0 if (p+r)==0 else 2*p*r*1.0 / (p + r)
        ret['NDCG@'+str(k)] = self.ndcg(k)

        return ret

    def hitratio(self, k):
        """ list : gt, pred=[(id, score)]
            return : float
        """
        hit = 0
        for single in self._results:
            user = single["user"]
            gt   = single["gt"]
            pred = sorted(zip(single["candidate"], single["score"]), key=lambda x: x[1], reverse=True)[:k]
            pred_id = [i[0] for i in pred]
            if len( set(gt) & set(pred_id) ) > 0 : hit += 1
            
        return hit * 1.0 / len(self._results)

    def precise(self, k):
        hit = 0 
        tot = 0 
        for single in self._results:
            user = single["user"]
            gt   = single["gt"]
            pred = sorted(zip(single["candidate"], single["score"]), key=lambda x: x[1], reverse=True)[:k]
            pred_id = [i[0] for i in pred]

            hit += len( set(gt) & set(pred_id) ) 
            tot += len(set( pred_id ))
        
        return hit * 1.0 / tot

    def recall (self, k):
        hit = 0 
        tot = 0
        for single in self._results:
            user = single["user"]
            gt   = single["gt"]
            pred = sorted(zip(single["candidate"], single["score"]), key=lambda x: x[1], reverse=True)[:k]
            pred_id = [i[0] for i in pred]

            hit += len( set(gt) & set(pred_id) ) 
            tot += len( set( gt ) )

        return hit * 1.0 / tot

    def ndcg   (self, k):
        import math
        tot_dcg = 0.0
        for single in self._results:
            idcg = 0.0
            dcg = 0.0
            ndcg = 0.0
            user = single["user"]
            gt   = single["gt"]
            pred = sorted(zip(single["candidate"], single["score"]), key=lambda x: x[1], reverse=True)[:k]
            pred_id = [i[0] for i in pred]
           
            for idx in range(min(len(pred), len(gt))):
                idcg += 1.0 / math.log(idx+2) * math.log(2)
            for idx, p in enumerate(pred_id):
                if p in gt : dcg += 1.0 / math.log(idx+2) * math.log(2)
            tot_dcg += dcg / idcg
        return tot_dcg / len(self._results)

    def dump(self, results):
        self._logger.info("Dump metric to {}".format(self._output_file))
        small_table = create_small_table(results)
        self._logger.info("Evaulation results for recommendation:\n" + small_table)
        
        with open(self._output_file, "w") as f:
            f.write("RecommendationEvaluator:\n" + small_table)
            f.write("\n\n")
            for k, v in results.items():
                f.write(str(k) + "\t\t" + str(v) + "\n")

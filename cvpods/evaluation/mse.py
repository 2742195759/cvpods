import copy
import itertools
import logging
import os
import copy
import numpy as np
from collections import OrderedDict

import torch

from cvpods.utils import PathManager, comm, create_small_table

from .evaluator import DatasetEvaluator
from .registry import EVALUATOR


@EVALUATOR.register()
class MSEEvaluator(DatasetEvaluator):
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
        self._output_file = os.path.join(cfg.OUTPUT_DIR, "metric_" + 
            dataset_name.split("_")[1])

    def reset(self):
        self._oup_score = []
        self._inp_score = []
    
    # @interface
    def process(self, inputs, outputs):
        """
        @param:
            inputs : [
                dict(
                    "user": int64
                    "gt"  : int64
                    "score": float32
                )
            ]
            outputs : 
                dict() {
                    "score": [float32]
                }

        """
        # print (type(outputs))
        # print (outputs)
        error_msg = "return in eval model is not satisfied prototype of RecommendationEvaluator: score -> [float64]"
        assert isinstance(outputs, dict)
        assert isinstance(outputs["score"], list), error_msg
        score = outputs['score']
        assert len(score) == len(inputs) and isinstance(outputs["score"][0], float), error_msg
        self._inp_score.extend([ inp['score']   for inp in inputs])
        self._oup_score.extend(score)
       
    def evaluate(self):
        return_dict = OrderedDict()
        mse = ((np.array(self._oup_score) - np.array(self._inp_score)) ** 2).mean()
        rmse = np.sqrt(((np.array(self._oup_score) - np.array(self._inp_score)) ** 2).mean())
        mae = np.abs(np.array(self._oup_score) - np.array(self._inp_score)).mean()
        return_dict['mse'] = mse
        return_dict['rmse'] = rmse
        return_dict['mae'] = mae

        if self._dump : 
            self.dump(return_dict)

        return return_dict

    def dump(self, results):
        self._logger.info("Dump metric to {}".format(self._output_file))
        small_table = create_small_table(results)
        self._logger.info("Evaulation results for mse:\n" + small_table)
        
        with open(self._output_file, "w") as f:
            f.write("MSE Evaluator:\n" + small_table)
            f.write("\n\n")
            for k, v in results.items():
                f.write(str(k) + "\t\t" + str(v) + "\n")


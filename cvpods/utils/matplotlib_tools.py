""" 
This file is used for matplotlib reuse code. 

xiongkun
2021-09-15
"""
import matplotlib.pyplot as plt
import os
import PIL
import numpy as np
from PIL import Image 

class MatplotlibBase:

    def set_mode(self, mode):
        assert mode in ['gui', 'png', 'data']

    def __init__(self):
        self.log_dir = './log/plt/output.jpg'
        self.tmp_file = '/var/log/plt-tmp.jpg'
        self.mode = 'data'
        pass

    def draw(self): 
        """ Draw on the given plt
            param: fig
                returned by plt.figure(num=0)
        """
        raise NotImplementedError()

    def __call__(self, *args, **kargs):
        self.fig = plt.figure(num=0)
        self.draw(*args, **kargs)
        ret = None
        if self.mode == 'data':
            plt.savefig(self.tmp_file)
            im = Image.open(self.tmp_file)
            arr = np.array(im)
            ret = arr
            ret = ret.transpose([2,0,1])

        elif self.mode == 'file':
            plt.savefig(self.log_dir)

        self.fig = None
        plt.clf()
        return ret

class PltHistogram(MatplotlibBase):
    def __init__(self):
        super(PltHistogram, self).__init__()
    def draw(self, tensor_list, bincount=70, prefix="", xylabels=["", ""], label=None):
        if not isinstance(tensor_list, list):
            tensor_list = [tensor_list]
        fig,ax=plt.subplots(figsize=(8,5))
        array_list = [ tensor.numpy() for tensor in tensor_list ]
        if xylabels[0]: plt.xlabel(xylabels[0])
        if xylabels[1]: plt.ylabel(xylabels[1])
        ax.set_title(prefix)
        if not label: 
            label = range(len(tensor_list)) 
        ax.hist(array_list, bincount, histtype='bar',label=label)
        ax.legend()

if __name__ == "__main__":
    _dict = dict(
        VISDOM=dict(
            HOST="192.168.1.1", 
            PORT="8082", 
            TURN_ON=True,
            ENV_PREFIX='test',
        ), 
    )

    import torch
    from easydict import EasyDict
    cfg = EasyDict(_dict)
    from cvpods.utils import (
        CommonMetricPrinter, JSONWriter, PathManager,
        TensorboardXWriter, collect_env_info, comm,
        seed_all_rng, setup_logger, VisdomWriter
    )
    from cvpods.utils import EventStorage
    with EventStorage() as storage:
        hist = PltHistogram()
        hist.mode = 'data'
        visdom_writer = VisdomWriter(cfg.VISDOM.HOST, cfg.VISDOM.PORT, 1, [], cfg.VISDOM.ENV_PREFIX)
        storage.put_image('histo', hist(torch.rand((1000,))))
        visdom_writer.write()




# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Model Zoo API for Detectron2: a collection of functions to create common model architectures and
optionally load pre-trained weights as released in
`MODEL_ZOO.md <https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md>`_.
"""
from .model_zoo import get, get_config, get_checkpoint_s3uri

__all__ = ['get', 'get_config', "get_checkpoint_s3uri"]

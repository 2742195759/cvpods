# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .citypersons import CityPersonsDataset
from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .crowdhuman import CrowdHumanDataset
from .imagenet import ImageNetDataset
from .lvis import LVISDataset
from .objects365 import Objects365Dataset
from .torchvision_datasets import CIFAR10Dataset, STL10Datasets
from .voc import VOCDataset
from .widerface import WiderFaceDataset
from .google_color import GoogleColorDataset
from .cifar import CIFARDataset
from .rec_amazon import AmazonDataset
from .ebay_color import EbayColorDataset
from .mnist import MNISTDataset
#from .referit import ReferitDataset
#from .referit_fast import ReferitFastDataset

__all__ = [
    "COCODataset",
    "VOCDataset",
    "CityScapesDataset",
    "ImageNetDataset",
    "WiderFaceDataset",
    "LVISDataset",
    "CityPersonsDataset",
    "AmazonDataset",
    "AmazonDatasetSubstitution",
    #"ReferitDataset", 
    #"ReferitFastDataset", 
    "GoogleColorDataset",
    "EbayColorDataset",
    "CIFARDataset", 
    "Objects365Dataset",
    "CrowdHumanDataset",
    "CIFAR10Dataset",
    "STL10Datasets",
    "MNISTDataset",
]


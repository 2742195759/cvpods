# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os.path as osp
import torch
import logging

from cvpods.configs.base_config import config
from cvpods.checkpoint import Checkpointer
from cvpods.utils import dynamic_import

logger = logging.getLogger(__name__)


# Default oss prefix
_S3_PREFIX = "DEFAULT"
# Default playground path
_PLAYGROUND_PATH = "DEFAULT"


def _check_config_path(
    config_path: str,
    playground_path: str = _PLAYGROUND_PATH
):
    """
    Check whether the specified model exists in `cvpods_playground/examples`.

    Args:
        config_path (str): config file path relative to "cvpods_playground/" directory,
            e.g., "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x"
        playground_path (str): `cvpods_playground` absolute path,
            e.g., "/data/repos/cvpods_playground", default values is
            abspath("cvpods/cvpods_playground/").
    """
    config_abs_path = osp.join(playground_path, config_path, "config.py")
    if not osp.exists(config_abs_path):
        raise RuntimeError(
            "{} not available in Model Zoo!".format(config_abs_path))


def get_checkpoint_s3uri(
    config_path: str,
    playground_path: str = _PLAYGROUND_PATH,
    s3_prefix: str = _S3_PREFIX
):
    """
    Args:
        config_path (str): config file path relative to "cvpods_playground/" directory,
            e.g., "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x"
        playground_path (str): `cvpods_playground` absolute path,
            e.g., "/data/repos/cvpods_playground", default values is
            abspath("cvpods/cvpods_playground/")
        s3_prefix (str): the s3 prefix of the trained model in oss,
            e.g., "s3://cvpodsdumps/user/playground", default values is
            "s3://generalDetection/cvpods/model_zoo".

    Returns:
        str: a S3Uri to the trained model weight,
            e.g., "s3://generalDetection/cvpods/model_zoo/examples/detection/coco/faster_rcnn.res50.fpn.coco.multiscale.1x/model_final.pth" # noqa
    """
    _check_config_path(config_path, playground_path)
    return osp.join(s3_prefix, config_path, "model_final.pth")


def get_config(
    config_path: str,
    custom_config: str,
    playground_path: str = _PLAYGROUND_PATH,
):
    """
    Returns cvpods Config instance using the given model name.

    Args:
        config_path (str): config file path relative to "cvpods_playground/" directory,
            e.g., "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x"
        custom_config (dict): custom hyperparameter configuration dictionary.
        playground_path(str): `cvpods_playground` absolute path,
            e.g., "/data/repos/cvpods_playground", default values is
            abspath("cvpods/cvpods_playground/").

    Returns:
        cfg (Config): specified model's config.
    """
    _check_config_path(config_path, playground_path)

    config_abs_path = osp.join(playground_path, config_path)
    cfg = dynamic_import("config", config_abs_path).config
    if custom_config:
        cfg._register_configuration(custom_config)

    return cfg


def get(
    config_path: str,
    trained: bool = False,
    custom_config: dict = None,
    playground_path: str = _PLAYGROUND_PATH,
    s3_prefix: str = _S3_PREFIX,
):
    """
    Get a model specified by model name under cvpods_playground's `examples` directory.

    Args:
        config_path (str): config file path relative to "cvpods_playground/" directory,
            e.g., "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x"
        trained (bool): Whether to initialize with the trained model zoo weights. If False, the
            initialization weights specified in the config file's `MODEL.WEIGHTS` key are used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.
        custom_config (dict): custom hyperparameter configuration dictionary.
        playground_path (str): `cvpods_playground` absolute path,
            e.g., "/data/repos/cvpods_playground", default values is
            abspath("cvpods/cvpods_playground/").
        s3_prefix (str): the s3 prefix of the trained model in oss,
            e.g., "s3://cvpodsdumps/user/playground", default values is
            "s3://generalDetection/cvpods/model_zoo".

    Examples:
        >>> # Example 1:
        >>> from cvpods import model_zoo
        >>> model = model_zoo.get(
                "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x",
                trained=True
            )

        >>> # Example 2: customize some hyperparameters
        >>> custom_config = dict(
                DATALOADER=dict(
                    NUM_WORKERS=8,
                ),
                INPUT=dict(
                    FORMAT="RGB",
                ),
                # ...
            )
        >>> model = model_zoo.get(
                "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x",
                trained=False,
                custom_config,
            )

        >>> # Example 3: customize playground path and oss prefix
        >>> model = model_zoo.get(
                "examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x",
                trained=True,
                playground_path="/data/repos/cvpods_playground",
                s3_prefix="s3://cvpodsdumps/user/playground/"
            )
    """

    cfg = get_config(config_path, custom_config, playground_path)

    # example:
    # /data/repos/cvpods/cvpods_playground/examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x # noqa
    config_abs_path = osp.join(playground_path, config_path)
    build = dynamic_import("net", config_abs_path).build_model

    if trained:
        cfg.MODEL.WEIGHTS = get_checkpoint_s3uri(config_path, playground_path, s3_prefix)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    model = build(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    return model


if __name__ == "__main__":
    custom_config = dict(
        DATALOADER=dict(
            NUM_WORKERS=8,
        ),
        INPUT=dict(
            FORMAT="RGB",
        ),
    )
    model = get(
        config_path="examples/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x",
        trained=False,
        custom_config=custom_config,
        playground_path="/data/repos/cvpods_playground",
    )
    print(type(model))

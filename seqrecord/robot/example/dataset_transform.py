"""Implement modality transform utilities for habitat dataset."""

from dataclasses import dataclass
import sys
from typing import Any, Dict

import numpy as np
import torch
from torchvision import transforms


def instantiate_modal_class(args: Dict[str, Any]) -> Any:
    """Instantiates a class [defined in this module] with the given args that contains class name
    and init args.

    Args:
        init: Dict of the form {"modal":...,"$attributes":...}.
    Returns:
        The instantiated class object.
    """
    args_class = getattr(
        sys.modules[__name__], args["modal"]
    )  # get class object from this module by its class name
    return args_class(**args["kwargs"])


class InputTransform:
    def __init__(self, modal_config: Dict[str, Any]) -> None:
        self.inputs = {}
        self.feature_map = {}
        for key, modal in modal_config.items():
            modal["kwargs"]["feature"] = key
            if modal["kwargs"].get("feature_in_batch", None) is not None:
                # this feature is not being sent to model
                self.feature_map[key] = modal["kwargs"]["feature_in_batch"]
            self.inputs[key] = instantiate_modal_class(modal)
        return

    def pre_transform(self, x: Dict[str, np.ndarray]):
        """apply pre-transform on frame data.

        Args:
            x (Dict[str, np.ndarray]): each value of the dict is one single frame data
        """
        res = {}
        for key, value in x.items():
            # static methods can be called from instances:
            # https://docs.python.org/3/library/functions.html#staticmethod
            res[key] = self.inputs[key].pre_transform(value)
        return res

    def transform(
        self, transform_type: str, x: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:

        res = {}
        for _, input in self.inputs.items():
            key = input.feature
            if key not in self.feature_map:
                # this input is not being sent to model in batch
                continue

            res[self.feature_map[key]] = self.inputs[key].transform(
                transform_type, x[key]
            )
        return res


class RGBImage:
    """

    Args:
        ModalityTransform (): RGB images
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, feature: str, img_dim: int, feature_in_batch: str):
        self.feature: str = feature
        self.feature_in_batch: str = feature_in_batch
        # transforms.Compose is not supported by torch jit script.
        # see: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
        # center crop and resize: If the image is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
        # normalize: essential code is tensor.sub_(mean).div_(std), so it should be fine with 'sequence' (batched) data.
        # transforms.Compose is not supported by torch jit script.
        self.train_trans = torch.nn.Sequential(
            transforms.RandomResizedCrop(img_dim, scale=(0.8, 1.0)),
            transforms.Normalize(mean=RGBImage.MEAN, std=RGBImage.STD),
        )

    @staticmethod
    def pre_transform(x: np.ndarray) -> np.ndarray:
        """Transform numpy array of image (size:[h,w,c], dtype:int8) to array (size:[c, h, w],
        dtype:float32)

        Args:
            x (np.ndarray): (size:[h,w,c], dtype:int8)

        Returns:
            np.ndarray: image (size:[c, h, w], dtype:float32)
        """
        x = x.transpose(2, 0, 1)  # h, w, c -> c, h, w
        return x

    def transform(self, transform_type: str, x: np.ndarray) -> torch.Tensor:
        """Apply transform to batch or a single frame of images represented by torch.Tensor, not
        pil image!

        Args:
            transform_type (str): type of transform {train|val}
            x (np.ndarray): batch of sequence of rgb image represented by np.ndarry with shape  h, w, c

        Returns:
            np.ndarray: with shape [num_units, c, unit_len, h, w]
        """
        pass

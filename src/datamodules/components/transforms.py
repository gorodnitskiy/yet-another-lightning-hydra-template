from typing import Any

from omegaconf import DictConfig

import hydra
import numpy as np
import albumentations

from PIL import Image


class TransformsWrapper:
    def __init__(self, config: DictConfig) -> None:
        self.set_stage("train")
        # train augmentations
        train_aug = []
        if not config.train.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for aug_name in config.train.get("order"):
            aug = hydra.utils.instantiate(config.train.get(aug_name))
            train_aug.append(aug)
        self.train_aug = albumentations.Compose(train_aug)

        # valid and test augmentations
        valid_test_aug = []
        if not config.valid_test.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for aug_name in config.valid_test.get("order"):
            aug = hydra.utils.instantiate(config.valid_test.get(aug_name))
            valid_test_aug.append(aug)
        self.valid_test_aug = albumentations.Compose(valid_test_aug)

    @classmethod
    def set_stage(cls, stage: str) -> None:
        cls.stage = stage

    def __call__(self, image: Any, **kwargs) -> Any:
        if isinstance(image, Image.Image):
            image = np.asarray(image)
        if self.stage == "train":
            return self.train_aug(image=image, **kwargs)
        return self.valid_test_aug(image=image, **kwargs)

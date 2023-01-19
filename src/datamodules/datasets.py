import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.datamodules.components.dataset import BaseDataset
from src.datamodules.components.h5_file import H5PyFile
from src.datamodules.components.parse import parse_image_paths


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        json_path: Optional[str] = None,
        txt_path: Optional[str] = None,
        data_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
        shuffle_on_load: bool = True,
        label_type: str = "torch.LongTensor",
        **kwargs: Any,
    ) -> None:
        """ClassificationDataset.

        Args:
            json_path (:obj:`str`, optional): Path to annotation json.
            txt_path (:obj:`str`, optional): Path to annotation txt.
            data_path (:obj:`str`, optional): Path to HDF5 file or images source dir.
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
            shuffle_on_load (bool): Deterministically shuffle the dataset on load
                to avoid the case when Dataset slice contains only one class due to
                annotations dict keys order. Default to True.
            label_type (str): Label torch.tensor type. Default to torch.FloatTensor.
            kwargs (Any): Additional keyword arguments for H5PyFile class.
        """

        super().__init__(transforms, read_mode, to_gray)
        if (json_path and txt_path) or (not json_path and not txt_path):
            raise ValueError("Requires json_path or txt_path, but not both.")
        elif json_path:
            json_path = Path(json_path)
            if not json_path.is_file():
                raise RuntimeError(f"'{json_path}' must be a file.")
            with open(json_path) as json_file:
                self.annotation = json.load(json_file)
        else:
            txt_path = Path(txt_path)
            if not txt_path.is_file():
                raise RuntimeError(f"'{txt_path}' must be a file.")
            self.annotation = {}
            with open(txt_path) as txt_file:
                for line in txt_file:
                    _, label, path = line[:-1].split("\t")
                    self.annotation[path] = label

        self.keys = list(self.annotation)
        if shuffle_on_load:
            random.Random(shuffle_on_load).shuffle(self.keys)

        self.label_type = label_type

        data_path = "" if data_path is None else data_path
        self.data_path = data_path = Path(data_path)
        self.data_file = None
        if data_path.is_file():
            if data_path.suffix != ".h5":
                raise RuntimeError(f"'{data_path}' must be a h5 file.")
            self.data_file = H5PyFile(str(data_path), **kwargs)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        data_file = self.data_file
        if data_file is None:
            source = self.data_path / key
        else:
            source = data_file[key]
        image = self._read_image_(source)
        image = self._process_image_(image)
        label = torch.tensor(self.annotation[key]).type(self.label_type)
        return {"image": image.float(), "label": label, "name": key}

    def get_weights(self) -> List[float]:
        label_list = [self.annotation[key] for key in self.keys]
        weights = 1.0 / np.bincount(label_list)
        return weights.tolist()


class ClassificationVicRegDataset(ClassificationDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        data_file = self.data_file
        if data_file is None:
            source = self.data_path / key
        else:
            source = data_file[key]
        image = self._read_image_(source)
        image1, image2 = np.copy(image), np.copy(image)
        # albumentations returns random augmentation on each __call__
        z1 = self._process_image_(image1)
        z2 = self._process_image_(image2)
        label = torch.tensor(self.annotation[key]).type(self.label_type)
        return {
            "z1": z1.float(),
            "z2": z2.float(),
            "label": label,
            "name": key,
        }


class NoLabelsDataset(BaseDataset):
    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        dir_paths: Optional[List[str]] = None,
        txt_paths: Optional[List[str]] = None,
        json_paths: Optional[List[str]] = None,
        dirname: Optional[str] = None,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
    ) -> None:
        """NoLabelsDataset.

        Args:
            file_paths (:obj:`List[str]`, optional): List of files.
            dir_paths (:obj:`List[str]`, optional): List of directories.
            txt_paths (:obj:`List[str]`, optional): List of TXT files.
            json_paths (:obj:`List[str]`, optional): List of JSON files.
            dirname (:obj:`str`, optional): Images source dir.
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
        """

        super().__init__(transforms, read_mode, to_gray)
        if file_paths or dir_paths or txt_paths:
            self.keys = parse_image_paths(
                file_paths=file_paths, dir_paths=dir_paths, txt_paths=txt_paths
            )
        elif json_paths:
            self.keys = []
            for json_path in json_paths:
                with open(json_path) as json_file:
                    data = json.load(json_file)
                for path in data.keys():
                    self.keys.append(path)
        else:
            raise ValueError("Requires data_paths or json_paths.")
        self.dirname = Path(dirname if dirname else "")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        path = self.dirname / Path(key)
        image = self._read_image_(path)
        image = self._process_image_(image)
        return {"image": image, "name": key}

    def __len__(self) -> int:
        return len(self.keys)

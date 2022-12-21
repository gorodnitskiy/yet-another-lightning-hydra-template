import json
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from pathlib import Path

from src.datamodules.components.dataset import BaseDataset
from src.datamodules.components.parse import parse_image_paths
from src.datamodules.components.h5_file import H5PyFile


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        json_path: Optional[str] = None,
        lst_path: Optional[str] = None,
        data_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(transforms, read_mode, to_gray)
        if (json_path and lst_path) or (not json_path and not lst_path):
            raise ValueError("Requires json_path or lst_path, but not both.")
        elif json_path:
            json_path = Path(json_path)
            if not json_path.is_file():
                raise RuntimeError(f"'{json_path}' must be a file.")
            with open(json_path, "r") as json_file:
                self.annotation = json.load(json_file)
        else:
            lst_path = Path(lst_path)
            if not lst_path.is_file():
                raise RuntimeError(f"'{lst_path}' must be a file.")
            self.annotation = {}
            with open(lst_path, "r") as lst_file:
                for line in lst_file:
                    _, label, path = line[:-1].split("\t")
                    self.annotation[path] = label

        self.keys = list(self.annotation)

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
        """
        Do not open h5 file only once, for example in class initialization!
        PyTorch use lazy evaluations, so in this case multi-threading isn't
        capable and dataloader will be work with only 1 worker (file was
        opened only 1 time).
        For dealing with it, open h5 file in each method which forwards to
        h5 file content. In this case each dataloader worker can open
        h5 file regardless of class initialization.
        """
        key = self.keys[index]
        if self.data_file is None:
            source = self.data_path / key
        else:
            # copy class instance for current dataloader worker
            data_file = self.data_file
            source = data_file[key]
        image = self._read_image_(source)
        image = self._process_image_(image)
        label = self.annotation[key]
        label = torch.tensor(label).long() if label else None
        return {"image": image.float(), "label": label}

    def get_labels(self) -> List[Any]:
        return [self.annotation[key] for key in self.keys]


class ClassificationVicRegDataset(ClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        if self.data_file is None:
            source = self.data_path / key
        else:
            # copy class instance for current dataloader worker
            data_file = self.data_file
            source = data_file[key]
        image = self._read_image_(source)
        image1, image2 = np.copy(image), np.copy(image)
        # albumentations returns random augmentation on each __call__
        z1 = self._process_image_(image1)
        z2 = self._process_image_(image2)
        label = self.annotation[key]
        label = torch.tensor(label).long() if label else None
        return {"z1": z1.float(), "z2": z2.float(), "label": label}


class OnlyImagesDataset(BaseDataset):
    def __init__(
        self,
        image_paths: Optional[str] = None,
        dir_paths: Optional[str] = None,
        lst_paths: Optional[str] = None,
        json_paths: Optional[str] = None,
        dirname: Optional[str] = None,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
    ) -> None:
        super().__init__(transforms, read_mode, to_gray)
        if image_paths or dir_paths or lst_paths:
            self.keys = parse_image_paths(image_paths, dir_paths, lst_paths)
        elif json_paths:
            self.keys = []
            for json_path in json_paths:
                with open(json_path, "r") as json_file:
                    data = json.load(json_file)
                for path in data.keys():
                    self.keys.append(path)
        else:
            raise ValueError("Requires data_paths or json_paths.")
        self.dirname = Path(dirname if dirname else "")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.dirname / Path(self.keys[index])
        image = self._read_image_(path)
        image = self._process_image_(image)
        return {"image": image, "path": str(path), "label": None}

    def __len__(self) -> int:
        return len(self.keys)

    def get_labels(self) -> List[Any]:
        raise NotImplementedError()

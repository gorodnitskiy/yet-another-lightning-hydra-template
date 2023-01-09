import io
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
    ) -> None:
        self.read_mode = read_mode
        self.to_gray = to_gray
        self.transforms = transforms

    def _read_image_(self, image: Any) -> np.ndarray:
        if self.read_mode == "pillow":
            if not isinstance(image, (str, Path)):
                image = io.BytesIO(image)
            image = np.asarray(Image.open(image).convert("RGB"))
        elif self.read_mode == "cv2":
            if not isinstance(image, (str, Path)):
                image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError("use pillow or cv2")
        if self.to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def _process_image_(self, image: np.ndarray) -> torch.Tensor:
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return torch.from_numpy(image).permute(2, 0, 1)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

import os
from typing import List, Optional, Tuple


def parse_image_paths(
    image_paths: Optional[List[str]] = None,
    dir_paths: Optional[List[str]] = None,
    lst_paths: Optional[List[str]] = None,
    xts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> List[str]:
    img_paths = list()
    if image_paths:
        for image_path in image_paths:
            _, xt = os.path.splitext(image_path)
            if xt in xts:
                img_paths.append(image_path)

    if dir_paths:
        for dir_path in dir_paths:
            for d, dirs, files in os.walk(dir_path):
                for file in files:
                    _, xt = os.path.splitext(file)
                    if xt in xts:
                        img_paths.append(os.path.join(d, file))

    if lst_paths:
        for lst_path in lst_paths:
            with open(lst_path) as lst_file:
                for line in lst_file:
                    img_path = line[:-1].split("\t")[-1]
                    img_paths.append(img_path)

    if len(img_paths) == 0:
        raise ValueError(f"No images with extensions: {xts}")
    return img_paths

import os
from typing import List, Optional, Tuple


def parse_image_paths(
    file_paths: Optional[List[str]] = None,
    dir_paths: Optional[List[str]] = None,
    txt_paths: Optional[List[str]] = None,
    xts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> List[str]:
    """Parse image paths from List, List of directories or List of txt files.

    Args:
        file_paths (:obj:`List[str]`, optional): List of files.
        dir_paths (:obj:`List[str]`, optional): List of directories.
        txt_paths (:obj:`List[str]`, optional): List of TXT files.
        xts (Tuple[str, ...]): Image extensions which are allowed.
            Default to (".jpg", ".jpeg", ".png").

    Returns:
        List[str]: Parsed images paths.
    """

    paths = []
    if file_paths:
        for file_path in file_paths:
            _, xt = os.path.splitext(file_path)
            if xt in xts:
                paths.append(file_path)

    if dir_paths:
        for dir_path in dir_paths:
            for d, dirs, files in os.walk(dir_path):
                for file in files:
                    _, xt = os.path.splitext(file)
                    if xt in xts:
                        paths.append(os.path.join(d, file))

    if txt_paths:
        for txt_path in txt_paths:
            with open(txt_path) as txt_file:
                for line in txt_file:
                    path = line[:-1].split("\t")[-1]
                    paths.append(path)

    if not paths:
        raise ValueError(f"No images with extensions: {xts}")
    return paths

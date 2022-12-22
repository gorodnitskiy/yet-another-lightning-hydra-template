import os
import random
from typing import Any, Dict, Iterable

import numpy as np
import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
)

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)

nvmlInit()
GPUS_NUM = torch.cuda.device_count()
GPU_CARDS = (nvmlDeviceGetHandleByIndex(num) for num in range(GPUS_NUM))


def get_gpu_memory_info(cards: Iterable = GPU_CARDS) -> None:
    """Get GPU memory info by PYNVML for each GPU: total, free and used."""
    for i, card in enumerate(cards):
        info = nvmlDeviceGetMemoryInfo(card)
        log.info(f"GPU memory info: card {i} : total : {info.total}")
        log.info(f"GPU memory info: card {i} : free  : {info.free}")
        log.info(f"GPU memory info: card {i} : used  : {info.used}")


def set_seed(
    seed: int = 42, deterministic: bool = True, benchmark: bool = False
) -> None:
    """Manually set seeds, deterministic and benchmark modes.

    Included seeds:
        - random.seed
        - np.random.seed
        - torch.random.manual_seed
        - torch.cuda.manual_seed
        - torch.cuda.manual_seed_all

    Also, manually set up deterministic and benchmark modes.

    Args:
        seed (int): Seed. Default to 42.
        deterministic (bool): deterministic mode. Default to True.
        deterministic (bool): benchmark mode. Default to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.cuda.deterministic = deterministic
        torch.cuda.benchmark = benchmark


def set_max_threads(max_threads: int = 32) -> None:
    """Manually set max threads.

    Threads set up for:
    - OMP_NUM_THREADS
    - OPENBLAS_NUM_THREADS
    - MKL_NUM_THREADS
    - VECLIB_MAXIMUM_THREADS
    - NUMEXPR_NUM_THREADS

    Args:
        max_threads (int): Max threads value. Default to 32.
    """
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)


def collect_random_states() -> Dict[str, Any]:
    """Collect random states: random, numpy, torch, torch.cuda."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch.cuda": torch.cuda.random.get_rng_state_all(),
    }

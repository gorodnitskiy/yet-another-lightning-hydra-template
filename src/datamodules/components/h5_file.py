import concurrent.futures as futures
import itertools
import os
import queue
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

_H5PY_FILES_CACHE = weakref.WeakValueDictionary()


def parallel_generator(
    func: Callable,
    array: Iterable,
    n_jobs: Optional[int] = None,
    buffer: int = 1024,
) -> Any:
    """Generator in parallel threads."""
    array = iter(array)
    thread_queue = queue.Queue(buffer)
    n_jobs = os.cpu_count() if n_jobs is None else n_jobs
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Prefill thread queue with buffer elements
        for item in itertools.islice(array, buffer):
            thread_queue.put(executor.submit(func, item))
        # Start giving out results, while refilling tasks
        for item in array:
            yield thread_queue.get().result()
            thread_queue.put(executor.submit(func, item))
        # Give out remaining results
        while not thread_queue.empty():
            yield thread_queue.get().result()


class H5PyFile:
    """This is a wrapper around a h5py file/dataset, which discards the open
    dataset, when pickling and reopens it, when unpickling, instead of trying
    to pickle the h5py.File object itself.

    Please note, that this wrapper doesn't provide any extra safeguards
    for parallel interactions with the dataset. Reading in parallel is safe,
    writing in parallel may be not. Check h5py docs, when in doubt.

    Thanks to Andrei Stoskii for this module which I reworked slightly.
    """

    def __init__(self, filename: str, **kwargs) -> None:
        self.dataset = None
        self.filename = filename
        self._kwargs = kwargs

    def _lazy_load_(self) -> None:
        if self.dataset is not None:
            return

        can_use_cache = True
        if self._kwargs.get("mode", "r") != "r":
            # Non-read mode
            can_use_cache = False
        if {k for k in self._kwargs if k != "mode"}:
            # Non-standard extra kwargs
            can_use_cache = False

        # Load dataset (from cache or from disk)
        dataset = None
        if can_use_cache:
            dataset = _H5PY_FILES_CACHE.get(self.filename, None)
        if dataset is None:
            dataset = h5py.File(self.filename, swmr=True, **self._kwargs)

        # Save dataset to cache and to self
        if can_use_cache:
            _H5PY_FILES_CACHE[self.filename] = dataset
        self.dataset = dataset

    def __getitem__(self, *args, **kwargs) -> Any:
        self._lazy_load_()
        return self.dataset.__getitem__(*args, **kwargs)[...]

    def __setitem__(self, *args, **kwargs) -> Any:
        self._lazy_load_()
        return self.dataset.__setitem__(*args, **kwargs)

    def __getstate__(self) -> Tuple[str, Dict[str, Any]]:
        return self.filename, self._kwargs

    def __setstate__(self, state: Tuple[str, Dict[str, Any]]) -> Any:
        return self.__init__(state[0], **state[1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filename='{self.filename}')"

    @classmethod
    def create(
        cls,
        filename: str,
        content: List[str],
        dirname: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Create h5py file for dataset from scratch.

        Args:
            filename (str): Name for h5py file.
            content (List[str]): Dataset content. Requires List[data filepath].
            dirname (:obj:`str`, optional): Additional dirname for image filepaths.
                Default to None.
            verbose (bool): Verbose option. If True, it would show tqdm progress bar.
                Default to True.
        """
        filename = Path(filename)
        ext = filename.suffix
        if ext != ".h5":
            raise RuntimeError(
                f"Expected extension to be '.h5', instead got '{ext}'."
            )
        dirname = Path("" if dirname is None else dirname)
        progress_bar = tqdm if verbose else (lambda it, *_, **__: it)

        # Check that all files exist
        generator = parallel_generator(
            lambda fp: (dirname / fp, (dirname / fp).is_file()),
            content,
            n_jobs=128,
        )
        for filepath, found in progress_bar(
            generator, desc="Indexing content", total=len(content)
        ):
            if not found:
                raise FileNotFoundError(filepath)

        # Read files from disk and save them to the dataset
        generator = parallel_generator(
            lambda fp: (fp, np.fromfile(dirname / fp, dtype=np.uint8)),
            content,
            n_jobs=128,
        )
        with h5py.File(filename, mode="x") as dataset:
            for filepath, data in progress_bar(
                generator, desc="Creating dataset", total=len(content)
            ):
                dataset[str(filepath)] = data

from collections import OrderedDict
from typing import List, Optional, Union

import torch
from pytorch_lightning import LightningModule, Trainer

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def process_state_dict(
    state_dict: Union[OrderedDict, dict],
    symbols: int = 0,
    exceptions: Optional[Union[str, List[str]]] = None,
) -> OrderedDict:
    """Filter and map model state dict keys.

    Args:
        state_dict (Union[OrderedDict, dict]): State dict.
        symbols (int): Determines how many symbols should be cut in the
            beginning of state dict keys. Default to 0.
        exceptions (Union[str, List[str]], optional): Determines exceptions,
            i.e. substrings, which keys should not contain.
    """
    new_state_dict = OrderedDict()
    if exceptions:
        if isinstance(exceptions, str):
            exceptions = [exceptions]
    for key, value in state_dict.items():
        is_exception = False
        if exceptions:
            for exception in exceptions:
                if key.startswith(exception):
                    is_exception = True
        if not is_exception:
            new_state_dict[key[symbols:]] = value

    return new_state_dict


def save_state_dicts(
    trainer: Trainer,
    model: LightningModule,
    dirname: str,
    symbols: int = 6,
    exceptions: Optional[Union[str, List[str]]] = None,
) -> None:
    """Save model state dicts for last and best checkpoints.

    Args:
        trainer (Trainer): Lightning trainer.
        model (LightningModule): Lightning model.
        dirname (str): Saving directory.
        symbols (int): Determines how many symbols should be cut in the
            beginning of state dict keys. Default to 6 for cutting
            Lightning name prefix.
        exceptions (Union[str, List[str]], optional): Determines exceptions,
            i.e. substrings, which keys should not contain.  Default to [loss].
    """
    # save state dict for last checkpoint
    mapped_state_dict = process_state_dict(
        model.state_dict(), symbols=symbols, exceptions=exceptions
    )
    path = f"{dirname}/last_ckpt.pth"
    torch.save(mapped_state_dict, path)
    log.info(f"Last ckpt state dict saved to: {path}")

    # save state dict for best checkpoint
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path == "":
        log.warning("Best ckpt not found! Saving was cancelled ...")
        return

    best_ckpt_score = trainer.checkpoint_callback.best_model_score
    if best_ckpt_score is not None:
        prefix = str(best_ckpt_score.detach().cpu().item())
        prefix = prefix.replace(".", "_")
    else:
        log.warning("Best ckpt score not found! Use prefix <unknown>!")
        prefix = "unknown"
    model = model.load_from_checkpoint(best_ckpt_path)
    mapped_state_dict = process_state_dict(
        model.state_dict(), symbols=symbols, exceptions=exceptions
    )
    path = f"{dirname}/best_ckpt_{prefix}.pth"
    torch.save(mapped_state_dict, path)
    log.info(f"Best ckpt state dict saved to: {path}")

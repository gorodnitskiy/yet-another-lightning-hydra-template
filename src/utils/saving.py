import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Union

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
        log.warning("Best ckpt not found! Skipping...")
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


def save_predictions_from_dataloader(
    predictions: List[Any], path: Path
) -> None:
    """Save predictions returned by `Trainer.predict` method for single
    dataloader.

    Args:
        predictions (List[Any]): Predictions returned by `Trainer.predict` method.
        path (Path): Path to predictions.
    """
    if path.suffix == ".csv":
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file)
            for batch in predictions:
                keys = list(batch.keys())
                batch_size = len(batch[keys[0]])
                for i in range(batch_size):
                    row = {key: batch[key][i].tolist() for key in keys}
                    writer.writerow(row)

    elif path.suffix == ".json":
        processed_predictions = {}
        for batch in predictions:
            keys = [key for key in batch.keys() if key != "names"]
            batch_size = len(batch[keys[0]])
            for i in range(batch_size):
                item = {key: batch[key][i].tolist() for key in keys}
                if "names" in batch.keys():
                    processed_predictions[batch["names"][i]] = item
                else:
                    processed_predictions[len(processed_predictions)] = item
        with open(path, "w") as json_file:
            json.dump(processed_predictions, json_file, ensure_ascii=False)

    else:
        raise NotImplementedError(f"{path.suffix} is not implemented!")


def save_predictions(
    predictions: List[Any], dirname: str, output_format: str = "json"
) -> None:
    """Save predictions returned by `Trainer.predict` method.

    Due to `LightningDataModule.predict_dataloader` return type is
    Union[DataLoader, List[DataLoader]], so `Trainer.predict` method can return
    a list of dictionaries, one for each provided batch containing their
    respective predictions, or a list of lists, one for each provided dataloader
    containing their respective predictions, where each list contains dictionaries.

    Args:
        predictions (List[Any]): Predictions returned by `Trainer.predict` method.
        dirname (str): Dirname for predictions.
        output_format (str): Output file format. It could be `json` or `csv`.
            Default to `json`.
    """
    if not predictions:
        log.warning("Predictions is empty! Saving was cancelled ...")
        return

    if output_format not in ("json", "csv"):
        raise NotImplementedError(
            f"{output_format} is not implemented! Use `json` or `csv`."
            "Or change `src.utils.saving.save_predictions` func logic."
        )

    path = Path(dirname) / "predictions"
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(predictions[0], dict):
        target_path = path / f"predictions.{output_format}"
        save_predictions_from_dataloader(predictions, target_path)
        log.info(f"Saved predictions to: {str(target_path)}")
        return

    elif isinstance(predictions[0], list):
        for idx, predictions_idx in enumerate(predictions):
            if not predictions_idx:
                log.warning(
                    f"Predictions for DataLoader #{idx} is empty! Skipping..."
                )
                continue
            target_path = path / f"predictions_{idx}.{output_format}"
            save_predictions_from_dataloader(predictions_idx, target_path)
            log.info(
                f"Saved predictions for DataLoader #{idx} to: "
                f"{str(target_path)}"
            )
        return

    raise Exception(
        "Passed predictions format is not supported by default!\n"
        "Make sure that it is formed correctly! It requires as List[Dict[str, Any]] type"
        "in case of predict_dataloader returns DataLoader or List[List[Dict[str, Any]]]"
        "type in case of predict_dataloader returns List[DataLoader]!\n"
        "Or change `src.utils.saving.save_predictions` function logic."
    )

from typing import Any, Callable, Dict, List, Optional, Tuple
import copy
import datetime
import math
import os
import time

from pytorch_lightning.callbacks import ProgressBarBase


def n_lines(text: str) -> int:
    return text.count("\n") + 1


def text_color(
    style: Optional[int] = None, color: Optional[int] = None
) -> Tuple[str, str]:
    if color is None:
        color_code = 0
    else:
        color_code = 30 + color
    if style is None:
        style_code = 0
    else:
        style_code = style
    return (
        "\033[" + str(style_code) + ";" + str(color_code) + "m",
        "\033[" + str(0) + ";" + str(0) + "m",
    )


def format_status(inp: Any) -> Any:
    if isinstance(inp, dict):
        for key in inp:
            inp[key] = format_status(inp[key])
    if isinstance(inp, (list, tuple)):
        for index in range(len(inp)):
            inp[index] = format_status(inp[index])
    elif isinstance(inp, int):
        if abs(inp) > 10**6:
            return f"{inp:.3e}"
        else:
            return f"{inp:d}"
    elif isinstance(inp, float):
        if abs(inp) > 10**6:
            return f"{inp:.3e}"
        elif abs(inp) < 10**-6:
            return f"{inp:.3e}"
        else:
            return f"{inp:.6f}"
    return inp


def colorize_string(string: str, colors: List[Any], padding: int = 0) -> str:
    substrings = []
    last_index = 0
    for color in colors:
        index = color[0] + padding
        substrings.append(string[last_index:index])
        substrings.append(color[1])
        last_index = index
    substrings.append(string[last_index:])
    return "".join(substrings)


def view_status(inp: Any, display_len: int = 80) -> str:
    separator = " | "
    strings = [""]
    colors = [[]]
    color_index = 0

    max_len = 0
    for key in inp:
        max_len = max(len(str(key)), max_len)

    for key in inp:
        start, end = text_color(style=1, color=color_index + 1)
        colors[-1].append((len(strings[-1]), start))
        strings[-1] += ("{:>" + str(max_len) + "s} ").format(key)
        colors[-1].append((len(strings[-1]), end))

        if isinstance(inp[key], (list, tuple)):
            strings[-1] += separator.join(inp[key])
        elif isinstance(inp[key], dict):
            pos = len(strings[-1])
            sub_res = []
            for sub_key in inp[key]:
                start, end = text_color(style=3, color=color_index + 1)
                colors[-1].append((pos, start))
                colors[-1].append((pos + len(sub_key), end))
                sub_res.append(sub_key + ": " + str(inp[key][sub_key]))
                pos = (
                    pos
                    + len(sub_key)
                    + len(": ")
                    + len(str(inp[key][sub_key]))
                    + len(separator)
                )
            strings[-1] += separator.join(sub_res)
        else:
            strings[-1] += str(inp[key])

        strings.append("")
        colors.append([])

        color_index += 1
        color_index %= 6

    new_strings = ["=" * display_len]
    for index in range(len(strings)):
        string = strings[index]
        position = 0
        color_index = 0
        padding = 0
        while len(string) > 0:
            splitter_location = -1
            if len(string) > display_len:
                splitter_location = string[:display_len].rfind(" | ")
            split_colors = []
            if splitter_location > 0:
                string_end = splitter_location
            else:
                string_end = min(display_len, len(string))
            while (
                color_index < len(colors[index])
                and colors[index][color_index][0] - position
                < string_end - padding
            ):
                split_colors.append(list(colors[index][color_index]))
                split_colors[-1][0] -= position
                color_index += 1

            if len(string) < display_len:
                to_print = string
                to_print = to_print + " " * (display_len - len(to_print))
                new_strings.append(
                    colorize_string(to_print, split_colors, padding=padding)
                )
                break
            elif splitter_location > 0:
                to_print = string[:splitter_location]
                to_print = to_print + " " * (display_len - len(to_print))
                new_strings.append(
                    colorize_string(to_print, split_colors, padding=padding)
                )
                string = " " * (max_len + 1) + string[(splitter_location + 3):]
                position += splitter_location + 3 - padding
                padding = max_len + 1
            else:
                to_print = string[:string_end]
                to_print = to_print + " " * (display_len - len(to_print))
                new_strings.append(
                    colorize_string(to_print, split_colors, padding=padding)
                )
                string = " " * (max_len + 1) + string[string_end:]
                position += string_end - padding
                padding = max_len + 1

    new_strings.append("=" * display_len)
    return "\n".join(new_strings)


def dict_to_multi_dict(status: Dict[str, Any]) -> Dict[str, Any]:
    decomposed_status = {}
    for key in list(status.keys()):
        key_parts = key.split("/")
        if len(key_parts) > 2:
            continue
        if len(key_parts) > 1:
            super_key = key_parts[0]
            sub_key = "/".join(key_parts[1:])
            if super_key not in decomposed_status:
                decomposed_status[super_key] = {}
            decomposed_status[super_key][sub_key] = status[key]
        else:
            decomposed_status[key] = status[key]
    return decomposed_status


def get_width() -> int:
    try:
        return os.get_terminal_size()[0] - 2
    except Exception as exception:
        print(exception)
        pass
    return 100


class StageProgressBar(object):
    def __init__(
        self, width_function: Callable, display_id: str = f"ep{0}"
    ) -> None:
        self.width_function = width_function
        self.width = 0
        self.last_vals = None
        self.finalized = False
        self.started = False
        self.display_id = display_id

    def __str__(self) -> str:
        status = format_status(self.last_vals)
        to_view = view_status(
            dict_to_multi_dict(status), display_len=self.width
        )
        return to_view

    @staticmethod
    def display(content: str) -> None:
        print(content, end="")
        print("\033[" + str(n_lines(content)) + "A")

    def __del__(self) -> None:
        self.finalize()

    def update(self, vals: Any) -> None:
        if self.finalized:
            return
        self.width = self.width_function()
        self.last_vals = vals
        cur_info = str(self)
        if not self.started:
            self.started = True
        self.display(cur_info)

    def finalize(self) -> None:
        if not self.finalized and (self.last_vals is not None):
            print(str(self))


def progress_str(width: int, state: float) -> str:
    progress = width * state
    filled = int(math.floor(progress))

    if filled < width:
        remnant = str(int(math.floor((progress - filled) * 10.0)))
        return "[" + "=" * filled + remnant + " " * (width - filled - 1) + "]"
    else:
        return "[" + "=" * width + "]"


class TimeEstimator(object):
    def __init__(self, eta_threshold: float = 0.001) -> None:
        self.eta_threshold = eta_threshold
        self.start_time = time.time()
        self.cur_state = 0
        self.est_finish_time = None

    def reset(self) -> Any:
        self.start_time = time.time()
        self.cur_state = 0
        self.est_finish_time = None
        return self

    def update(self, cur_state: float) -> None:
        self.cur_state = cur_state
        if self.cur_state >= self.eta_threshold:
            self.est_finish_time = (
                self.start_time
                + (time.time() - self.start_time) / self.cur_state
            )

    def __str__(self) -> str:
        elapsed = str(
            datetime.timedelta(seconds=int(time.time() - self.start_time))
        )
        if self.est_finish_time is not None:
            eta = str(
                datetime.timedelta(
                    seconds=int(self.est_finish_time - time.time())
                )
            )
        else:
            eta = "?"
        return f"[{elapsed}>{eta}]"


class LiteProgressBar(ProgressBarBase):
    def __init__(self) -> None:
        super().__init__()
        self.last_epoch = 0
        self.pbar = StageProgressBar(
            width_function=get_width, display_id=f"ep{0}"
        )
        self.timer = TimeEstimator()
        self.display_counter = 0
        self.enable = True

    def disable(self) -> None:
        self.enable = False

    def enable(self) -> None:
        self.enable = True

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        self.timer.reset()
        trainer = args[0]
        log = copy.deepcopy(trainer.logged_metrics)
        if "epoch" in log:
            log["Info/epoch"] = copy.deepcopy(log["epoch"])
            del log["epoch"]
        log["Info/Mode"] = "train"
        log["Info/Progress"] = progress_str(15, 0)
        log["Info/Time"] = str(self.timer)
        self.pbar.update(log)
        self.pbar.update(trainer.logged_metrics)

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        trainer = args[0]
        log = copy.deepcopy(trainer.logged_metrics)
        if "epoch" in log:
            log["Info/epoch"] = copy.deepcopy(log["epoch"])
            del log["epoch"]
        log["Info/Mode"] = "train"
        log["Info/Progress"] = progress_str(15, 1.0)
        log["Info/Time"] = str(self.timer)
        self.pbar.update(log)
        self.pbar.update(trainer.logged_metrics)

    def step(
        self, part: str, batch_idx: int, total_batches: int, *args
    ) -> None:
        self.timer.update(float(batch_idx) / float(total_batches))
        trainer = args[0]
        log = copy.deepcopy(trainer.logged_metrics)
        if "epoch" in log:
            log["Info/epoch"] = copy.deepcopy(log["epoch"])
            del log["epoch"]
        log["Info/Mode"] = part
        log["Info/Progress"] = (
            progress_str(15, float(batch_idx) / float(total_batches))
            + f" {str(batch_idx)} / {str(total_batches)}"
        )
        log["Info/Time"] = str(self.timer)
        self.pbar.update(log)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        super().on_train_batch_end(*args, **kwargs)
        self.step(
            "train", self.train_batch_idx, self.total_train_batches, *args
        )

    def on_validation_epoch_start(self, *args, **kwargs) -> None:
        self.timer.reset()
        trainer = args[0]
        log = trainer.logged_metrics
        if "epoch" in log:
            log["Info/epoch"] = copy.deepcopy(log["epoch"])
            del log["epoch"]
        log["Info/Mode"] = "val"
        log["Info/Progress"] = progress_str(15, 0)
        log["Info/Time"] = str(self.timer)
        self.pbar.update(log)
        self.pbar.update(trainer.logged_metrics)

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        trainer = args[0]
        log = copy.deepcopy(trainer.logged_metrics)
        if "epoch" in log:
            log["Info/epoch"] = copy.deepcopy(log["epoch"])
            del log["epoch"]
        log["Info/Mode"] = "val"
        log["Info/Progress"] = progress_str(15, 1.0)
        log["Info/Time"] = str(self.timer)
        self.pbar.update(log)
        self.pbar.update(trainer.logged_metrics)

    def on_validation_batch_end(self, *args, **kwargs) -> None:
        super().on_validation_batch_end(*args, **kwargs)
        self.step(
            "val", self.val_batch_idx, self.total_val_batches, *args
        )

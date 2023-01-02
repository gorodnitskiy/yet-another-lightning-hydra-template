from pathlib import Path
from shutil import copytree
from subprocess import (  # nosec B404, B603
    STDOUT,
    SubprocessError,
    check_output,
)
from typing import Optional

from omegaconf import DictConfig


def run(cmd, allow_fail=True, no_env=True):
    """Run shell command by subprocess."""
    try:
        output = check_output(
            cmd,
            stderr=STDOUT,
            text=True,
            shell=True,  # nosec B604
            env={} if no_env else None,
        )
    except SubprocessError as exception:
        if allow_fail:
            output = f"{exception}\n\n{exception.output}"
        else:
            raise
    return f"> {cmd}\n\n{output}\n"


def log_pip_metadata(path: Path) -> None:
    """Collect pip metadata."""
    (path / "pip.log").write_text(
        "\n".join(
            [
                run("pip freeze --disable-pip-version-check"),
                run("pip freeze --disable-pip-version-check --user"),
            ]
        )
    )


def log_git_metadata(path: Path) -> None:
    """Collect git metadata."""
    (path / "git.log").write_text(
        "\n".join(
            [
                run("git describe --tags --long --dirty --always"),
                run("git describe --all --long --dirty --always"),
                run("git branch --verbose --verbose --all"),
                run("git remote --verbose"),
                run("git status"),
            ]
        )
    )


def log_gpu_metadata(path: Path) -> None:
    """Collect GPU metadata."""
    (path / "gpu.log").write_text(
        "\n".join(
            [
                run("env | grep -E '(NV|CU)' | sort", no_env=False),
                run("nvidia-smi"),
            ]
        )
    )


def log_metadata(cfg: DictConfig) -> None:
    """Log pip, git and GPU metadata.

    Save code and configs folders as artifacts.
    """
    target_path = Path(cfg.paths.output_dir) / "metadata"
    target_path.mkdir(parents=True, exist_ok=True)

    # Logging pip, git and GPU metadata
    log_pip_metadata(target_path)
    log_git_metadata(target_path)
    log_gpu_metadata(target_path)

    # Saving code and configs folders
    script_path = Path(cfg.paths.root_dir) / "src"
    if not script_path.is_dir():
        raise RuntimeError("Couldn't find the src folder.")
    target_src_path = target_path / "src"
    if not target_src_path.exists():
        copytree(script_path, target_src_path)

    configs_path = Path(cfg.paths.root_dir) / "configs"
    if not configs_path.is_dir():
        raise RuntimeError("Couldn't find the configs folder.")
    target_configs_path = target_path / "configs"
    if not target_configs_path.exists():
        copytree(configs_path, target_configs_path)

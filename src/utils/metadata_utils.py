import subprocess  # nosec B404, B603
from pathlib import Path
from shutil import copytree
from typing import Any

from omegaconf import DictConfig


def run_sh_command(
    cmd: Any, allow_fail: bool = True, no_env: bool = True, **kwargs: Any
) -> str:
    """Run shell command by subprocess."""
    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,  # nosec B604
            env={} if no_env else None,
            **kwargs,
        )
    except subprocess.SubprocessError as exception:
        if allow_fail:
            output = f"{exception}\n\n{exception.output}"
        else:
            raise
    return f"> {cmd}\n\n{output}\n"


def log_pip_metadata(path: Path) -> None:
    """Collect pip metadata."""
    outputs = [
        run_sh_command("pip freeze --disable-pip-version-check"),
        run_sh_command("pip freeze --disable-pip-version-check --user"),
    ]
    (path / "pip.log").write_text("\n".join(outputs))


def log_git_metadata(path: Path) -> None:
    """Collect git metadata."""
    outputs = [
        run_sh_command("git describe --tags --long --dirty --always"),
        run_sh_command("git describe --all --long --dirty --always"),
        run_sh_command("git branch --verbose --verbose --all"),
        run_sh_command("git remote --verbose"),
        run_sh_command("git status"),
    ]
    (path / "git.log").write_text("\n".join(outputs))


def log_gpu_metadata(path: Path) -> None:
    """Collect GPU metadata."""
    outputs = [
        run_sh_command("env | grep -E '(NV|CU)' | sort", no_env=False),
        run_sh_command("nvidia-smi"),
    ]
    (path / "gpu.log").write_text("\n".join(outputs))


def log_metadata(cfg: DictConfig) -> None:
    """Log pip, git and GPU metadata. Save code and configs folders as
    artifacts.

    Args:
        cfg (DictConfig): Main config.
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

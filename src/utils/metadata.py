from pathlib import Path
from omegaconf import DictConfig
from shutil import copytree

import subprocess


def run(cmd, allow_fail=True, no_env=True):
    """Run shell command by subprocess."""
    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
            env={} if no_env else None,
        )
    except subprocess.SubprocessError as exception:
        if allow_fail:
            output = f"{exception}\n\n{exception.output}"
        else:
            raise
    return f"> {cmd}\n\n{output}\n"


def pip_metadata(path: Path) -> None:
    """Collect pip metadata."""
    (path / "pip_metadata.txt").write_text(
        "\n".join(
            [
                run(f"pip freeze --disable-pip-version-check"),
                run(f"pip freeze --disable-pip-version-check --user"),
            ]
        )
    )


def git_metadata(path: Path) -> None:
    """Collect git metadata."""
    (path / "git_metadata.txt").write_text(
        "\n".join(
            [
                run(f"git describe --tags --long --dirty --always"),
                run(f"git describe --all --long --dirty --always"),
                run(f"git branch --verbose --verbose --all"),
                run(f"git remote --verbose"),
                run(f"git status"),
            ]
        )
    )


def gpu_metadata(path: Path) -> None:
    """Collect GPU metadata."""
    (path / "gpu_metadata.txt").write_text(
        "\n".join(
            [
                run("env | grep -E '(NV|CU)' | sort", no_env=False),
                run("nvidia-smi"),
            ]
        )
    )


def save_code_as_artifact(cfg: DictConfig) -> None:
    """Save code and configs folders as artifacts."""
    script_path = Path(cfg.paths.root_dir) / "src"
    if not script_path.is_dir():
        raise RuntimeError("Couldn't find the src folder.")
    target_path = Path(cfg.paths.output_dir) / "src"
    if not target_path.exists():
        copytree(script_path, target_path)

    configs_path = Path(cfg.paths.root_dir) / "configs"
    if not configs_path.is_dir():
        raise RuntimeError("Couldn't find the configs folder.")
    target_path = Path(cfg.paths.output_dir) / "configs"
    if not target_path.exists():
        copytree(configs_path, target_path)


def log_metadata(cfg: DictConfig) -> None:
    """Logging pip, git and GPU metadata. Save code and configs folders."""
    path = Path(cfg.paths.output_dir)
    pip_metadata(path)
    git_metadata(path)
    gpu_metadata(path)
    save_code_as_artifact(cfg)

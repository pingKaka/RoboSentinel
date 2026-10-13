#!/usr/bin/env python3
"""Sync data and orchestrate Qwen3-VL fine-tuning over SSH."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = PROJECT_ROOT / "training/config/remote.env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("check", "sync", "setup", "prepare", "train", "status", "download"),
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    return parser.parse_args()


def load_env(path: Path) -> dict[str, str]:
    env_path = path.expanduser().resolve()
    if not env_path.is_file():
        raise FileNotFoundError(
            f"Missing {env_path}. Copy remote.env.example to remote.env and edit it."
        )
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid environment line {env_path}:{line_number}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    required = {
        "REMOTE_HOST",
        "REMOTE_USER",
        "REMOTE_PORT",
        "REMOTE_WORKDIR",
        "REMOTE_CONDA_SH",
        "REMOTE_CONDA_ENV",
        "LOCAL_OUTPUT_DIR",
    }
    missing = sorted(key for key in required if not values.get(key))
    if missing:
        raise ValueError(f"Missing remote configuration values: {missing}")
    return values


def ssh_base(config: dict[str, str]) -> list[str]:
    command = ["ssh", "-p", config["REMOTE_PORT"]]
    identity = config.get("REMOTE_IDENTITY_FILE", "")
    if identity:
        command.extend(("-i", str(Path(identity).expanduser())))
    command.append(f"{config['REMOTE_USER']}@{config['REMOTE_HOST']}")
    return command


def remote_shell(config: dict[str, str], body: str, check: bool = True) -> int:
    command = ssh_base(config) + ["bash", "-lc", shlex.quote(body)]
    return subprocess.run(command, check=check).returncode


def activated_command(config: dict[str, str], command: str) -> str:
    return " && ".join(
        (
            f"source {shlex.quote(config['REMOTE_CONDA_SH'])}",
            f"conda activate {shlex.quote(config['REMOTE_CONDA_ENV'])}",
            f"cd {shlex.quote(config['REMOTE_WORKDIR'])}",
            command,
        )
    )


def sync(config: dict[str, str]) -> int:
    remote_shell(config, f"mkdir -p {shlex.quote(config['REMOTE_WORKDIR'])}")
    ssh_transport = ["ssh", "-p", config["REMOTE_PORT"]]
    identity = config.get("REMOTE_IDENTITY_FILE", "")
    if identity:
        ssh_transport.extend(("-i", str(Path(identity).expanduser())))
    destination = (
        f"{config['REMOTE_USER']}@{config['REMOTE_HOST']}:"
        f"{config['REMOTE_WORKDIR'].rstrip('/')}/training/"
    )
    command = [
        "rsync",
        "-az",
        "--info=progress2",
        "--exclude",
        "config/remote.env",
        "--exclude",
        "output/",
        "--exclude",
        "downloaded_output/",
        "-e",
        shlex.join(ssh_transport),
        str(PROJECT_ROOT / "training") + "/",
        destination,
    ]
    return subprocess.run(command, check=True).returncode


def download(config: dict[str, str]) -> int:
    destination = Path(config["LOCAL_OUTPUT_DIR"]).expanduser()
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    ssh_transport = ["ssh", "-p", config["REMOTE_PORT"]]
    identity = config.get("REMOTE_IDENTITY_FILE", "")
    if identity:
        ssh_transport.extend(("-i", str(Path(identity).expanduser())))
    source = (
        f"{config['REMOTE_USER']}@{config['REMOTE_HOST']}:"
        f"{config['REMOTE_WORKDIR'].rstrip('/')}/training/output/"
    )
    command = [
        "rsync",
        "-az",
        "--info=progress2",
        "-e",
        shlex.join(ssh_transport),
        source,
        str(destination) + "/",
    ]
    return subprocess.run(command, check=True).returncode


def main() -> int:
    args = parse_args()
    config = load_env(args.env)
    if args.action == "sync":
        return sync(config)
    if args.action == "download":
        return download(config)
    if args.action == "check":
        body = activated_command(
            config,
            "python --version && nvidia-smi --query-gpu=name,memory.total,memory.free "
            "--format=csv,noheader && (swift --version || true) && df -h .",
        )
        return remote_shell(config, body)
    if args.action == "setup":
        body = activated_command(
            config,
            "python -m pip install -r training/requirements.txt",
        )
        return remote_shell(config, body)
    if args.action == "prepare":
        body = activated_command(
            config,
            "python training/run/prepare_dataset.py --config training/config/data.yaml "
            "&& python training/run/validate_dataset.py --config training/config/data.yaml",
        )
        return remote_shell(config, body)
    if args.action == "train":
        training = (
            "mkdir -p training/output && "
            "(nohup python training/run/run_training.py --config training/config/train.yaml "
            "> training/output/train.log 2>&1 < /dev/null & "
            "echo $! > training/output/train.pid) && "
            "echo Started PID $(cat training/output/train.pid)"
        )
        return remote_shell(config, activated_command(config, training))
    if args.action == "status":
        status = (
            "if [ -f training/output/train.pid ]; then "
            "pid=$(cat training/output/train.pid); "
            "if kill -0 $pid 2>/dev/null; then echo RUNNING PID=$pid; "
            "else echo STOPPED PID=$pid; fi; "
            "else echo NO_PID_FILE; fi; "
            "tail -n 80 training/output/train.log 2>/dev/null || true; "
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader"
        )
        return remote_shell(config, activated_command(config, status))
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())

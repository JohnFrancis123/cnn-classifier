from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config: #Project defaults for training and serving.
    dataset: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 64
    epochs: int = 80
    lr: float = 0.001
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    num_workers: int = 2


def load_config(path: str | Path) -> Config: #Merge YAML overrides onto Config defaults.
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}
    return Config(**raw_config)
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as T
from pydantic import BaseModel, validator


class TrainerConfig(BaseModel):
    batch_size: int
    epochs: int
    epoch_size: int
    log_every_n_steps: int
    num_workers: int


class TrainingConfig(BaseModel):
    optimizer: torch.optim.Optimizer | Any
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | Any | None = None
    loss_function: nn.Module | Any

    class Config:
        arbitrary_types_allowed = True


class TransformConfig(BaseModel):
    training: T.Compose | Any
    validation: T.Compose | Any

    class Config:
        arbitrary_types_allowed = True


class DataConfig(BaseModel):
    train_path: Path
    validation_path: Path
    images_path: Path
    log_dir: Path

    @validator("train_path", "images_path", "log_dir")
    def validate_path(cls, v: str) -> Path:
        path = Path(v)
        if not path.exists():
            raise FileExistsError(f"{str(path)} does not exist.")
        return path


class MiscConfig(BaseModel):
    random_seed: int | None


class DummyRunConfig(BaseModel):
    dummy_run: bool
    limit_train_batches: int
    limit_val_batches: int


class Config(BaseModel):
    data: DataConfig
    misc: MiscConfig
    dummy_run: DummyRunConfig
    transforms: TransformConfig
    trainer: TrainerConfig
    training: TrainingConfig
    model: nn.Module | Any

    class Config:
        arbitrary_types_allowed = True

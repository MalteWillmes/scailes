from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class ImageDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        img_dir: str | Path,
        transform: Compose | None = None,
    ):
        self.metadata = metadata_df
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.class_map = {"Wild": 0, "Farmed": 1}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[Image.Image | Tensor, int]:
        img_path = self.img_dir / self.metadata.iloc[idx]["file_name"]
        image = Image.open(img_path)
        label = self.class_map[self.metadata.iloc[idx]["classification"]]
        if self.transform:
            image = self.transform(image)
        return image, label

    def train_validation_split(
        self, frac: float = 0.8, seed: int | None = None
    ) -> tuple[ImageDataset, ImageDataset]:
        """Split the dataset into a training and test set."""
        if seed is not None:
            np.random.seed(seed)
        mask = np.random.rand(len(self.metadata)) < frac
        train_df = pd.DataFrame.copy(self.metadata[mask])
        test_df = pd.DataFrame.copy(self.metadata[~mask])

        return (
            ImageDataset(train_df, self.img_dir, self.transform),
            ImageDataset(test_df, self.img_dir, self.transform),
        )

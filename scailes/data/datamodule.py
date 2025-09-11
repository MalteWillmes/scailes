import lightning as L
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler

from scailes.data.image_dataset import ImageDataset
from scailes.utils import make_weighted_sampler


class FishScaleDataModule(L.LightningDataModule):
    """Lightning data module."""

    def __init__(self, data_module_config: DictConfig):
        """
        Args:
            data_dir (Path): The directory where the data is stored.
            batch_size (int): The size of the batches for the DataLoader.
            train_epoch_size (int): The size of a training epoch.
        """
        super().__init__()
        self.config = data_module_config

        self.data_loader_config = self.config.data_loader
        self.transforms = self.config.transforms
        self.paths = self.config.paths

        self.val_class_counts: np.ndarray | None = None
        self.test_class_counts: np.ndarray | None = None

    def setup(self, stage: str) -> None:
        """
        Set up the data for training, validation, and testing.
        Args:
            stage (str): The stage of the model (training, validation, testing).
        """
        if stage == "fit":
            training_data = pd.read_csv(self.paths.train_path)
            validation_data = pd.read_csv(self.paths.validation_path)

            self.val_class_counts = (
                validation_data["classification"].value_counts().values  # type: ignore
            )

            self.train_set = ImageDataset(
                metadata_df=training_data,
                img_dir=self.paths.images_path,
                transform=self.transforms.training,
            )
            self.val_set = ImageDataset(
                metadata_df=validation_data,
                img_dir=self.paths.images_path,
                transform=self.transforms.validation,
            )

        elif stage == "test":
            test_data = pd.read_csv(self.paths.test_path)
            self.test_class_counts = (
                test_data["classification"].value_counts().values  # type: ignore
            )
            self.test_set = ImageDataset(
                metadata_df=test_data,
                img_dir=self.paths.images_path,
                transform=self.transforms.test,
            )

    def _create_dataloader(
        self, dataset: ImageDataset, sampler: WeightedRandomSampler | None = None
    ) -> DataLoader:
        """
        Create a DataLoader for a given dataset.
        Args:
            dataset: The dataset for which to create the DataLoader.
            sampler: The sampler to use for the DataLoader. Defaults to None.
        Return:
            DataLoader: The DataLoader for the given dataset.
        """
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            **self.data_loader_config,
        )

    def train_dataloader(self) -> DataLoader:
        """Create a DataLoader for the training data, using weighted sampler."""
        # Weighted sampler to balance wild/farmed:
        weighted_sampler = make_weighted_sampler(self.train_set, self.config.epoch_size)
        return self._create_dataloader(self.train_set, sampler=weighted_sampler)

    def val_dataloader(self) -> DataLoader:
        """Create a DataLoader for the validation data."""
        return self._create_dataloader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        """Create a DataLoader for the test data."""
        return self._create_dataloader(self.test_set)

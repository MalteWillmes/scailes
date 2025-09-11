import pandas as pd
import torch
import torch.utils.data as data
from torchvision.transforms import v2

from scailes.data.image_dataset import ImageDataset


def image_transform() -> v2.Compose:
    """Image transforms for the Transfer Leaning of the ResNext model."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((384, 512)),
            v2.RandomRotation(360),
            v2.Grayscale(num_output_channels=3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extended_image_transform() -> v2.Compose:
    """Same as image_transform, but with added HorizontalFlip and ColorJitter
    augmentations."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((384, 512)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            v2.RandomRotation(360),
            v2.Grayscale(num_output_channels=3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def autoaugment_image_transform() -> v2.Compose:
    """Image transform with only AutoAugment for the ResNext model."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((384, 512)),
            v2.Grayscale(num_output_channels=3),
            v2.AutoAugment(policy=v2.AutoAugmentPolicy.IMAGENET),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def minimal_transform_resnext(
    *, resize: int | tuple[int, int] | None = None
) -> v2.Compose:
    """Minimal transform required for ResNext model.
    See https://pytorch.org/hub/pytorch_vision_resnext/ for more details."""
    resize = resize or (384, 512)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(resize),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_weighted_sampler(
    dataset: ImageDataset, epoch_size: int
) -> data.WeightedRandomSampler:
    """Create a weighted sampler for the dataset based on the class distribution."""
    metadata: pd.DataFrame = dataset.metadata
    class_weights = (
        len(metadata) / metadata["classification"].value_counts()
    ).to_dict()
    sample_weights = metadata["classification"].apply(lambda x: class_weights[x]).values

    return data.WeightedRandomSampler(
        weights=sample_weights, num_samples=epoch_size, replacement=False
    )

import argparse

import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from scailes.models.classifier import ScailesClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the model.")
    parser.add_argument("checkpoint", type=str, help="The model checkpoint to test.")
    parser.add_argument(
        "hparams",
        type=str,
        help="The hyperparameter file for the model checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_file = args.checkpoint
    hparams_file = args.hparams

    # loading train config from the checkpoint
    config = OmegaConf.load(hparams_file)

    # using local config for the data module
    local_config = OmegaConf.load("scailes/config/config.yaml")
    config.data_module = local_config.data_module
    config.trainer.devices = local_config.trainer.devices

    tb_logger: TensorBoardLogger = instantiate(config.logger)
    seed_everything(config.misc.random_seed, workers=True)

    torch.set_float32_matmul_precision(config.misc.matmul_precision)

    data_module = instantiate(config.data_module)

    model = ScailesClassifier.load_from_checkpoint(
        checkpoint_path=checkpoint_file, config=config
    )

    trainer = instantiate(config.trainer, logger=tb_logger)
    trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

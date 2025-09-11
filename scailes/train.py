import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from scailes.models.classifier import ScailesClassifier


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    tb_logger: TensorBoardLogger = instantiate(config.logger)
    tb_logger.log_hyperparams(dict(config))
    tb_logger.save()
    seed_everything(config.misc.random_seed, workers=True)

    torch.set_float32_matmul_precision(config.misc.matmul_precision)

    data_module = instantiate(config.data_module)

    model = ScailesClassifier(config)

    trainer = instantiate(config.trainer, logger=tb_logger)

    if config.dummy_run.dummy_run:  # limit the number of batches for a dummy run
        trainer.limit_train_batches = config.dummy_run.limit_train_batches
        trainer.limit_val_batches = config.dummy_run.limit_val_batches

    # Fit the models
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

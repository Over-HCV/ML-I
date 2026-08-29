"""
Pieces shared by every data provider

Batching and class weighting depend on the labels and the configuration, not
on what a sample is, so both providers use the same implementation.
"""

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from training.imbalance import loss_weighting, resolve_strategy


def class_weighting(
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    train_labels: list[int],
    class_names: list[str],
) -> tuple[torch.Tensor | None, Any]:
    """Class weights and a balancing sampler for the configured strategy.

    Returns (weights, sampler); either may be None.
    """
    strategy = resolve_strategy(dataset_config, training_config)

    return loss_weighting(
        strategy, dataset_config, training_config, train_labels, class_names
    )


def make_loaders(
    datasets: dict[str, Dataset],
    training_config: dict[str, Any],
    sampler: Any = None,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Wrap train/val/test datasets in DataLoaders.

    The sampler applies to the training split only; validation and test are read
    in a fixed order so their metrics are comparable across epochs.
    """
    batch_size = training_config.get("batch_size", 32)

    # pin_memory is a CUDA feature and is not supported on MPS
    use_pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )
    }

    for split in ("val", "test"):
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )

    return loaders

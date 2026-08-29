"""
Class imbalance strategies

The Dataset page lets the user choose how to handle imbalance. Until now that
choice was written into the config and read by nothing - only the Training
page's own class_weights field affected a run. This module is what makes the
Dataset page's choice actually take effect.

Resampling applies to the training split only. Resampling validation or test
rows would invent samples that the model is then scored on, which makes every
reported metric meaningless.
"""

from typing import Any

import numpy as np
import torch

from training.dataset import compute_class_weights, create_weighted_sampler

AUTO_WEIGHTS = "Auto Class Weights"
MANUAL_WEIGHTS = "Manual Class Weights"
SMOTE_OVERSAMPLING = "Oversampling (SMOTE)"
UNDERSAMPLING = "Undersampling"
SELECTIVE_AUGMENTATION = "Selective Augmentation"
NO_ADJUSTMENT = "No Adjustment"

# Strategies that change the training rows rather than the loss
RESAMPLING_STRATEGIES = (SMOTE_OVERSAMPLING, UNDERSAMPLING)


def resolve_strategy(
    dataset_config: dict[str, Any], training_config: dict[str, Any]
) -> str:
    """Work out which imbalance strategy applies to this run.

    The Dataset page's choice wins. The Training page's older class_weights
    field is the fallback so existing configs keep behaving as they did.
    """
    strategy = dataset_config.get("imbalance_handling", {}).get("strategy", "")

    if strategy:
        # The radio labels carry suffixes like "(Recommended)" and "(H2)"
        for known in (
            AUTO_WEIGHTS,
            MANUAL_WEIGHTS,
            SMOTE_OVERSAMPLING,
            UNDERSAMPLING,
            SELECTIVE_AUGMENTATION,
            NO_ADJUSTMENT,
        ):
            if strategy.startswith(known):
                return known

    fallback = training_config.get("class_weights", "None")

    return AUTO_WEIGHTS if fallback == AUTO_WEIGHTS else fallback


def loss_weighting(
    strategy: str,
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    train_labels: list[int],
    class_names: list[str],
) -> tuple[torch.Tensor | None, Any]:
    """Class weights and an optional balancing sampler for a strategy.

    Returns (weights, sampler); either may be None.
    """
    num_classes = len(class_names)

    if strategy == AUTO_WEIGHTS:
        weights = compute_class_weights(train_labels, num_classes)
        print(
            f"[Imbalance] Auto class weights "
            f"(range: {weights.min():.2f} - {weights.max():.2f})"
        )
        return weights, create_weighted_sampler(train_labels, num_classes)

    if strategy == MANUAL_WEIGHTS:
        configured = dataset_config.get("imbalance_handling", {}).get("class_weights")
        if configured:
            weights = torch.tensor(
                [float(configured.get(name, 1.0)) for name in class_names],
                dtype=torch.float32,
            )
            print(f"[Imbalance] Manual class weights: {weights.tolist()}")
            return weights, None

    if training_config.get("class_weights") == "Focal Loss":
        print("[Imbalance] Focal Loss with class weights")
        return compute_class_weights(train_labels, num_classes), None

    return None, None


def resample(
    strategy: str,
    features: np.ndarray,
    labels: list[int],
    dataset_config: dict[str, Any],
    random_seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Oversample or undersample the training rows.

    Only meaningful for tabular features: SMOTE interpolates between feature
    vectors, which produces nonsense when the vector is an image.
    """
    if strategy not in RESAMPLING_STRATEGIES:
        return features, labels

    before = _counts(labels)

    try:
        features, labels = _apply_resampler(
            strategy, features, labels, dataset_config, random_seed
        )
    except Exception as exception:
        # A failed resample must not lose the run: fall back to the real rows.
        print(f"[Imbalance] {strategy} skipped: {exception}")
        return features, labels

    print(f"[Imbalance] {strategy}: {before} -> {_counts(labels)}")

    return features, labels


def _apply_resampler(
    strategy: str,
    features: np.ndarray,
    labels: list[int],
    dataset_config: dict[str, Any],
    random_seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Run the chosen sampler over the training rows"""
    if strategy == SMOTE_OVERSAMPLING:
        from imblearn.over_sampling import SMOTE

        ratio = dataset_config.get("imbalance_handling", {}).get("smote_ratio")
        # SMOTE builds synthetic points from a minority sample's neighbours, so
        # it needs at least a couple of neighbours to interpolate between.
        smallest_class = min(_counts(labels).values())
        neighbours = max(min(5, smallest_class - 1), 1)

        sampler = SMOTE(
            sampling_strategy=_sampling_ratio(ratio, labels),
            k_neighbors=neighbours,
            random_state=random_seed,
        )
    else:
        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(random_state=random_seed)

    resampled_features, resampled_labels = sampler.fit_resample(features, labels)

    return resampled_features, list(resampled_labels)


def _sampling_ratio(ratio: float | None, labels: list[int]) -> Any:
    """Translate the UI's ratio slider into a SMOTE sampling_strategy.

    The slider means "raise each minority class to this fraction of the
    majority". A ratio below a class's existing share would ask SMOTE to remove
    samples, which it cannot do, so those classes are left alone.
    """
    if ratio is None:
        return "auto"

    counts = _counts(labels)
    majority = max(counts.values())
    target = int(majority * ratio)

    return {
        label: max(count, target) for label, count in counts.items()
    }


def _counts(labels: list[int]) -> dict[int, int]:
    """Sample count per class label"""
    unique, counts = np.unique(np.asarray(labels), return_counts=True)

    return {int(label): int(count) for label, count in zip(unique, counts, strict=True)}

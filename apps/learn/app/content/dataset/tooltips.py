"""
Tooltips for dataset configuration page.
Centralized tooltip text for tabs, sections, and controls.
"""

# Tab-level tooltips
TAB_TOOLTIPS = {
    "overview": (
        "View dataset statistics, configure train/validation/test split ratios, "
        "and set up class imbalance handling strategies."
    ),
    "distribution": (
        "Visualize class distribution with charts. "
        "Select which malware families to include in training."
    ),
    "samples": (
        "Preview sample images from each class. "
        "Configure preprocessing (resize, normalization, color mode)."
    ),
    "augmentation": (
        "Configure data augmentation transforms applied during training. "
        "Preview augmentation effects on real samples."
    ),
}

# Section tooltips
SECTION_TOOLTIPS = {
    "dataset_overview": (
        "Basic statistics about your malware image dataset. "
        "Shows total samples, number of classes, and class balance ratio."
    ),
    "data_split": (
        "Configure how data is divided for training, validation, and testing. "
        "Stratified splits maintain class proportions in each subset."
    ),
    "cross_validation": (
        "K-Fold Cross-Validation trains K models, each using a different fold as validation. "
        "More robust evaluation but K times slower. Use for final model selection."
    ),
    "class_imbalance": (
        "Malware datasets often have unequal class sizes. "
        "Imbalance handling prevents the model from ignoring minority classes."
    ),
    "class_selection": (
        "Select which malware families to include in training. "
        "Excluding rare classes can improve model performance on remaining classes."
    ),
    "preprocessing": (
        "Image preprocessing applied before training. "
        "All images are resized to the same dimensions and normalized."
    ),
    "augmentation_config": (
        "Data augmentation creates variations of training images. "
        "Helps prevent overfitting and improves generalization."
    ),
}

# Specific control tooltips (for widgets without inline help)
CONTROL_TOOLTIPS = {
    "imbalance_auto_weights": (
        "Automatically adjusts loss function to weight minority classes higher. "
        "Weight = max_samples / class_samples. No data modification needed."
    ),
    "imbalance_selective_aug": (
        "Applies more aggressive augmentation to minority classes only. "
        "Tests H2 hypothesis: whether augmentation improves minority recall."
    ),
    "imbalance_smote": (
        "SMOTE interpolates between samples to create synthetic data. "
        "NOT recommended for images - creates blurred/unrealistic samples."
    ),
    "stratified_split": (
        "Ensures each split (train/val/test) has the same class proportions. "
        "Critical for imbalanced datasets to prevent validation bias."
    ),
    "random_seed": (
        "Fixed seed ensures reproducible splits across runs. "
        "Same seed = same train/val/test assignment for each sample."
    ),
    "augmentation_preset_none": "No augmentation. Use only for debugging or overfitting analysis.",
    "augmentation_preset_light": "Minimal augmentation: flips and 90° rotations only.",
    "augmentation_preset_moderate": "Balanced augmentation with color adjustments.",
    "augmentation_preset_heavy": "Aggressive augmentation including noise. May slow training.",
    "augmentation_preset_custom": "Fine-grained control over each augmentation type.",
    "orthogonal_rotation": (
        "Rotations by 90°, 180°, 270° only. "
        "Lossless (no interpolation artifacts) unlike arbitrary angle rotation."
    ),
}

# Imbalance strategy descriptions (for the radio buttons)
IMBALANCE_STRATEGIES = {
    "auto_weights": {
        "name": "Auto Class Weights (Recommended)",
        "description": (
            "Automatically calculates class weights inversely proportional to frequency. "
            "Classes with fewer samples get higher loss weights during training."
        ),
    },
    "selective_aug": {
        "name": "Selective Augmentation (H2)",
        "description": (
            "Applies stronger augmentation to minority classes only. "
            "Tests whether targeted augmentation improves minority class recall."
        ),
    },
    "manual_weights": {
        "name": "Manual Class Weights",
        "description": (
            "Manually specify importance weight for each class. "
            "Use when you have domain knowledge about class importance."
        ),
    },
    "smote": {
        "name": "Oversampling (SMOTE)",
        "description": (
            "Creates synthetic samples by interpolating between existing ones. "
            "NOT recommended for images - produces unrealistic blended pixels."
        ),
    },
    "undersample": {
        "name": "Undersampling",
        "description": (
            "Removes samples from majority classes to balance dataset. "
            "Simple but loses potentially valuable training data."
        ),
    },
    "none": {
        "name": "No Adjustment",
        "description": (
            "Train with natural class distribution. "
            "Model may become biased toward majority classes."
        ),
    },
}

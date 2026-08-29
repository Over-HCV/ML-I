"""
Dataset Tab 4: Augmentation & Configuration
Data augmentation settings and final configuration save
"""

import random

from PIL import Image, ImageEnhance
import streamlit as st

from content.dataset.tooltips import CONTROL_TOOLTIPS, SECTION_TOOLTIPS
from state.workflow import has_dataset_config, save_dataset_config
from utils.dataset_registry import DatasetEntry
from utils.dataset_utils import calculate_split_percentages


def render(dataset_info, entry: DatasetEntry):
    """Render augmentation settings and configuration save"""
    _init_augmentation_defaults()
    augmentation_config = render_augmentation_config()
    render_augmentation_preview(dataset_info, augmentation_config)
    st.divider()
    render_configuration_summary(dataset_info, augmentation_config, entry)


def _init_augmentation_defaults():
    """Initialize augmentation defaults if not in session state.

    This must run BEFORE widgets render so widgets read from session_state.
    """
    defaults = {
        "augmentation_preset": "Custom",
        "aug_h_flip": True,
        "aug_v_flip": True,
        "aug_rotation": True,
        "aug_rotation_angles": [90, 180, 270],
        "aug_brightness": True,
        "aug_brightness_range": 10,
        "aug_contrast": False,
        "aug_contrast_range": 10,
        "aug_noise": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_augmentation_config():
    """Data augmentation preset selection and custom config"""
    st.subheader("Data Augmentation", help=SECTION_TOOLTIPS["augmentation_config"])

    st.info(
        "Augmentation is applied during training, not during dataset preparation"
    )

    preset = st.radio(
        "Augmentation Preset",
        ["None", "Light", "Moderate", "Heavy", "Custom"],
        horizontal=True,
        key="augmentation_preset",
    )

    preset_info = {
        "None": "No augmentation applied",
        "Light": "Horizontal flip + Orthogonal rotation (90°/180°/270°)",
        "Moderate": "H/V flip + Orthogonal rotation + Brightness (±10%)",
        "Heavy": "All flips + Orthogonal rotation + Brightness/Contrast (±20%) + Noise",
    }

    if preset != "Custom":
        st.info(preset_info.get(preset, ""))

    # Store augmentation config
    augmentation_config = {"preset": preset}

    if preset == "Custom":
        st.markdown("**Custom Augmentation Configuration**")

        col1, col2 = st.columns(2)
        with col1:
            h_flip = st.checkbox(
                "Horizontal Flip",
                key="aug_h_flip",
                help="Flip image left-right. Safe for malware images.",
            )
            v_flip = st.checkbox(
                "Vertical Flip",
                key="aug_v_flip",
                help="Flip image top-bottom.",
            )
            rotation = st.checkbox(
                "Orthogonal Rotation",
                key="aug_rotation",
                help=CONTROL_TOOLTIPS["orthogonal_rotation"],
            )
            rotation_angles = st.session_state.get("aug_rotation_angles", [90, 180, 270])
            if rotation:
                rotation_angles = st.multiselect(
                    "Allowed Rotations",
                    options=[90, 180, 270],
                    key="aug_rotation_angles",
                    help="Only 90° multiples are lossless (no interpolation)",
                )

        with col2:
            brightness = st.checkbox(
                "Brightness Adjustment",
                key="aug_brightness",
            )
            brightness_range = st.session_state.get("aug_brightness_range", 10)
            if brightness:
                brightness_range = st.slider(
                    "Brightness Range (%)", 0, 50, key="aug_brightness_range"
                )

            contrast = st.checkbox(
                "Contrast Adjustment",
                key="aug_contrast",
            )
            contrast_range = st.session_state.get("aug_contrast_range", 10)
            if contrast:
                contrast_range = st.slider(
                    "Contrast Range (%)", 0, 50, key="aug_contrast_range"
                )

            noise = st.checkbox("Gaussian Noise", key="aug_noise")

        # Build custom config
        augmentation_config["custom"] = {
            "horizontal_flip": h_flip,
            "vertical_flip": v_flip,
            "rotation": rotation,
            "rotation_angles": rotation_angles if rotation else [],
            "brightness": brightness,
            "brightness_range": brightness_range if brightness else 0,
            "contrast": contrast,
            "contrast_range": contrast_range if contrast else 0,
            "gaussian_noise": noise,
        }

    return augmentation_config


def apply_preprocessing(img):
    """Apply preprocessing settings from session state"""
    target_size = st.session_state.get("preprocessing_target_size", "224x224")
    color_mode = st.session_state.get("preprocessing_color_mode", "RGB")

    result = img.copy()

    # Resize
    size = int(target_size.split("x")[0])
    result = result.resize((size, size), Image.Resampling.LANCZOS)

    # Color mode
    if color_mode == "Grayscale":
        result = result.convert("L")
    else:
        result = result.convert("RGB")

    return result


def render_augmentation_preview(dataset_info, augmentation_config):
    """Preview augmentation effects on a sample image"""
    st.divider()
    st.subheader("Augmentation Preview")

    if not dataset_info.get("sample_paths"):
        st.warning("No samples to preview")
        return

    # Family selector
    selected_family = st.selectbox(
        "Select Family to Preview",
        options=dataset_info["classes"],
        key="augmentation_preview_family",
    )

    sample_path = dataset_info["sample_paths"][selected_family][0]
    original = Image.open(sample_path)
    preprocessed = apply_preprocessing(original)

    # Show preprocessed and augmented versions
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**After Preprocessing**")
        st.image(preprocessed, width="stretch")

    with col2:
        st.markdown("**Augmented (Example 1)**")
        aug1 = apply_augmentation(preprocessed, augmentation_config)
        st.image(aug1, width="stretch")

    with col3:
        st.markdown("**Augmented (Example 2)**")
        aug2 = apply_augmentation(preprocessed, augmentation_config)
        st.image(aug2, width="stretch")

    # Refresh button to see different random augmentations
    if st.button("Generate New Examples", key="refresh_augmentation"):
        st.rerun()


def apply_augmentation(img, config):
    """Apply augmentation transforms to an image"""
    result = img.copy()

    if config.get("preset") == "None":
        return result

    # Get custom config or use preset defaults
    if config.get("preset") == "Custom":
        custom = config.get("custom", {})
    else:
        # Preset defaults
        presets = {
            "Light": {
                "horizontal_flip": True,
                "rotation": True,
                "rotation_angles": [90, 180, 270],
            },
            "Moderate": {
                "horizontal_flip": True,
                "vertical_flip": True,
                "rotation": True,
                "rotation_angles": [90, 180, 270],
                "brightness": True,
                "brightness_range": 10,
            },
            "Heavy": {
                "horizontal_flip": True,
                "vertical_flip": True,
                "rotation": True,
                "rotation_angles": [90, 180, 270],
                "brightness": True,
                "brightness_range": 20,
                "contrast": True,
                "contrast_range": 20,
            },
        }
        custom = presets.get(config.get("preset"), {})

    # Apply transforms randomly
    if custom.get("horizontal_flip") and random.random() > 0.5:
        result = result.transpose(Image.FLIP_LEFT_RIGHT)

    if custom.get("vertical_flip") and random.random() > 0.5:
        result = result.transpose(Image.FLIP_TOP_BOTTOM)

    if custom.get("rotation") and custom.get("rotation_angles"):
        angle = random.choice(custom.get("rotation_angles"))
        angles = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
        result = result.transpose(angles[angle])

    if custom.get("brightness"):
        factor = (
            1
            + random.uniform(
                -custom.get("brightness_range", 10), custom.get("brightness_range", 10)
            )
            / 100
        )
        result = ImageEnhance.Brightness(result).enhance(factor)

    if custom.get("contrast"):
        factor = (
            1
            + random.uniform(
                -custom.get("contrast_range", 10), custom.get("contrast_range", 10)
            )
            / 100
        )
        result = ImageEnhance.Contrast(result).enhance(factor)

    return result


def render_configuration_summary(dataset_info, augmentation_config, entry: DatasetEntry):
    """Final configuration summary and save button"""
    st.subheader("Configuration Summary")

    # Get split configuration
    use_cross_validation = st.session_state.get("use_cross_validation", False)

    # Use selected classes from session state, or all classes if none selected
    if "selected_classes" in st.session_state and st.session_state.selected_classes:
        selected_families = sorted(st.session_state.selected_classes)
    else:
        selected_families = sorted(dataset_info["classes"])

    # Calculate totals for selected classes only
    total_samples = sum(dataset_info["samples"].get(c, 0) for c in selected_families)

    # Get imbalance handling strategy
    imbalance_strategy = st.session_state.get(
        "imbalance_strategy", "Auto Class Weights (Recommended)"
    )
    class_weights = (
        st.session_state.get("class_weights", None)
        if imbalance_strategy == "Manual Class Weights"
        else None
    )

    # Build configuration based on split method
    if use_cross_validation:
        n_folds = st.session_state.get("n_folds", 5)
        split_config = {
            "method": "cross_validation",
            "n_folds": n_folds,
            "stratified": st.session_state.get("stratified_kfold", True),
            "random_seed": st.session_state.get("random_seed", 73),
        }
    else:
        train_pct = st.session_state.get("train_split", 70)
        val_of_remaining = st.session_state.get("val_split", 50)
        train_final, val_final, test_final = calculate_split_percentages(
            train_pct, val_of_remaining
        )

        split_config = {
            "method": "fixed_split",
            "train": round(train_final, 1),
            "val": round(val_final, 1),
            "test": round(test_final, 1),
            "stratified": st.session_state.get("stratified_split", True),
            "random_seed": st.session_state.get("random_seed", 73),
        }

    config = {
        "dataset_name": entry["name"],
        # Absolute so the training worker and the interpretability engine resolve
        # the same directory regardless of the process working directory.
        "dataset_path": str(entry["path"]),
        "total_samples": total_samples,
        "num_classes": len(selected_families),
        "selected_families": selected_families,
        "split": split_config,
        "augmentation": augmentation_config,
        "preprocessing": {
            "target_size": (224, 224),
            "normalization": "[0,1]",
            "color_mode": "RGB",
        },
        "imbalance_handling": {
            "strategy": imbalance_strategy,
            "class_weights": class_weights,
            "smote_ratio": st.session_state.get("smote_ratio", 0.5)
            if imbalance_strategy == "Oversampling (SMOTE)"
            else None,
            "selective_augmentation": {
                "enabled": imbalance_strategy == "Selective Augmentation (H2)",
                "threshold": st.session_state.get("minority_threshold", 200),
                "multiplier": st.session_state.get("aug_multiplier", 2.0),
                "minority_classes": st.session_state.get("minority_classes", []),
            }
            if imbalance_strategy == "Selective Augmentation (H2)"
            else None,
        },
    }

    # Show key info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected Classes", len(selected_families))
    with col2:
        st.metric("Total Samples", config["total_samples"])
    with col3:
        if use_cross_validation:
            st.metric("Method", f"{n_folds}-Fold CV")
        else:
            st.metric("Augmentation", augmentation_config["preset"])

    # Save button FIRST (before expander)
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if st.button("Save Configuration", type="primary", width="stretch"):
            save_dataset_config(config)
            st.rerun()

    if has_dataset_config():
        st.success("Configuration saved. Continue on the Model page.")

    # Display config with better formatting (collapsed by default)
    with st.expander("View Full Configuration", expanded=False):
        st.json(config)

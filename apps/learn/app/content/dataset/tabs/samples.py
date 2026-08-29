"""
Dataset Tab 3: Samples & Preprocessing
Sample image viewer and preprocessing preview
"""

import random

from PIL import Image
import streamlit as st


def render(dataset_info):
    """Render sample viewer and preprocessing preview"""
    _init_samples_defaults()
    render_preprocessing_preview(dataset_info)
    st.divider()
    render_sample_viewer(dataset_info)


def _init_samples_defaults():
    """Initialize samples/preprocessing defaults if not in session state.

    This must run BEFORE widgets render so widgets read from session_state.
    """
    defaults = {
        "preprocessing_target_size": "224x224",
        "preprocessing_normalization": "[0,1] Scale",
        "preprocessing_color_mode": "RGB",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sample_viewer(dataset_info):
    """Sample image browser - stable across reruns"""
    st.subheader("Dataset Samples")

    if not dataset_info["sample_paths"]:
        st.warning("No sample images found")
        return

    # Cache sample selection in session state to prevent re-randomizing on rerun
    if "gallery_samples" not in st.session_state:
        all_samples = []
        for paths in dataset_info["sample_paths"].values():
            all_samples.extend(paths[:6])  # More samples per family
        st.session_state.gallery_samples = random.sample(
            all_samples, min(36, len(all_samples))  # 6x6 grid
        )

    sample_paths = st.session_state.gallery_samples

    # Display 6x6 grid with fixed width to prevent overflow
    cols = st.columns(6)
    for idx, img_path in enumerate(sample_paths):
        with cols[idx % 6]:
            try:
                img = Image.open(img_path)
                st.image(img, width=150)
                st.caption(f"{img.size[0]}x{img.size[1]}")
            except Exception as e:
                st.error(f"Error: {img_path.name}")


def render_preprocessing_preview(dataset_info):
    """Preview with family selector"""
    st.subheader("Preprocessing Preview")

    if not dataset_info["sample_paths"]:
        st.warning("No sample images found")
        return

    # Family selector (affects the preview below)
    selected_family = st.selectbox(
        "Select Family to Preview",
        options=dataset_info["classes"],
        key="preprocessing_family_select",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        target_size = st.selectbox(
            "Target Size",
            ["224x224", "256x256", "299x299", "512x512"],
            key="preprocessing_target_size",
        )
    with col2:
        normalization = st.radio(
            "Normalization",
            ["[0,1] Scale", "[-1,1] Scale", "ImageNet Mean/Std"],
            key="preprocessing_normalization",
        )
    with col3:
        color_mode = st.radio(
            "Color Mode",
            ["RGB", "Grayscale"],
            key="preprocessing_color_mode",
        )

    # Use selected family's image
    if selected_family not in dataset_info["sample_paths"]:
        st.warning(f"No samples for {selected_family}")
        return

    sample_path = dataset_info["sample_paths"][selected_family][0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        try:
            original = Image.open(sample_path)
            st.image(original, width="stretch")
            st.caption(f"Size: {original.size[0]}x{original.size[1]}")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

    with col2:
        st.markdown("**After Preprocessing**")
        try:
            size = int(target_size.split("x")[0])
            processed = original.copy()
            processed = processed.resize((size, size), Image.Resampling.LANCZOS)

            if color_mode == "Grayscale":
                processed = processed.convert("L")
            elif color_mode == "RGB":
                processed = processed.convert("RGB")

            st.image(processed, width="stretch")
            st.caption(f"Size: {size}x{size}, Mode: {color_mode}")
        except Exception as e:
            st.error(f"Error processing: {e}")

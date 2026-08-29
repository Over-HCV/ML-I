"""
Dataset Tab 2: Class Distribution
Visualizes malware family distribution and allows class selection
"""

import plotly.graph_objects as go
import streamlit as st


def render(dataset_info, data_kind: str = "image_folder"):
    """Render class distribution visualizations with selection interface"""
    st.subheader("Class Distribution & Selection")

    if not dataset_info["samples"]:
        st.warning("No data found")
        return

    # Get all classes
    all_classes = sorted(dataset_info["samples"].keys())

    # Initialize session state for selected classes
    if "selected_classes" not in st.session_state:
        st.session_state.selected_classes = all_classes.copy()  # Select all by default

    # Drop any class that does not belong to the dataset on screen: Streamlit
    # raises if a multiselect default is missing from its options.
    st.session_state.selected_classes = [
        name for name in st.session_state.selected_classes if name in all_classes
    ] or all_classes.copy()

    # Class selection interface
    st.markdown("### Select Classes to Include")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Select All", width="stretch"):
            st.session_state.selected_classes = all_classes.copy()
            st.rerun()

    with col2:
        if st.button("Deselect All", width="stretch"):
            st.session_state.selected_classes = []
            st.rerun()

    with col3:
        # Quick filter for large/small classes
        threshold = st.number_input(
            "Min samples per class",
            min_value=0,
            max_value=1000,
            value=0,
            help="Filter classes with fewer samples than this threshold",
        )

        if threshold > 0:
            filtered_classes = [
                cls
                for cls in all_classes
                if dataset_info["samples"].get(cls, 0) >= threshold
            ]
            if st.button(f"Select classes with ≥{threshold} samples", width="stretch"):
                st.session_state.selected_classes = filtered_classes
                st.rerun()

    # Multi-select for individual class selection
    selected = st.multiselect(
        "Selected Classes (you can search by typing)",
        options=all_classes,
        default=st.session_state.selected_classes,
        key="class_selector",
        help="Select which malware families to include in your dataset",
    )

    # Update session state
    st.session_state.selected_classes = selected

    # Show selection summary
    if selected:
        total_samples = sum(dataset_info["samples"].get(c, 0) for c in selected)

        # Get split info
        use_cross_validation = st.session_state.get("use_cross_validation", False)

        if use_cross_validation:
            n_folds = st.session_state.get("n_folds", 5)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Classes", len(selected))
            with col2:
                st.metric("Total Samples", f"{total_samples:,}")
            with col3:
                st.metric("K-Folds", n_folds)
        else:
            from utils.dataset_utils import calculate_split_percentages

            train_pct = st.session_state.get("train_split", 70)
            val_of_remaining = st.session_state.get("val_split", 50)
            train_final, val_final, test_final = calculate_split_percentages(
                train_pct, val_of_remaining
            )

            train_samples = int(total_samples * train_final / 100)
            val_samples = int(total_samples * val_final / 100)
            test_samples = int(total_samples * test_final / 100)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Selected Classes", len(selected))
            with col2:
                st.metric("Total Samples", f"{total_samples:,}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Train ({train_final:.1f}%)", f"{train_samples:,}")
            with col2:
                st.metric(f"Val ({val_final:.1f}%)", f"{val_samples:,}")
            with col3:
                st.metric(f"Test ({test_final:.1f}%)", f"{test_samples:,}")
    else:
        st.warning("No classes selected. Select at least one class.")

    st.divider()

    # Visualization of distribution
    st.markdown("### Distribution Visualization")

    # Use selected classes for visualization
    display_classes = selected if selected else all_classes
    sample_counts = [dataset_info["samples"].get(c, 0) for c in display_classes]

    # Get split configuration from session state
    use_cross_validation = st.session_state.get("use_cross_validation", False)

    if use_cross_validation:
        # Cross-Validation: Show total samples per class
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Total Samples (used in CV)",
                x=display_classes,
                y=sample_counts,
                marker_color="#98c127",
                text=sample_counts,
                textposition="outside",
            )
        )

        n_folds = st.session_state.get("n_folds", 5)
        title_text = f"Samples per Malware Family - {n_folds}-Fold Cross-Validation"
    else:
        # Fixed split: Show Train/Val/Test breakdown
        from utils.dataset_utils import calculate_split_percentages

        train_pct = st.session_state.get("train_split", 70)
        val_of_remaining = st.session_state.get("val_split", 50)
        train_final, val_final, test_final = calculate_split_percentages(
            train_pct, val_of_remaining
        )

        # Calculate samples per class for each split
        train_counts = [
            int(dataset_info["samples"].get(c, 0) * train_final / 100)
            for c in display_classes
        ]
        val_counts = [
            int(dataset_info["samples"].get(c, 0) * val_final / 100)
            for c in display_classes
        ]
        test_counts = [
            int(dataset_info["samples"].get(c, 0) * test_final / 100)
            for c in display_classes
        ]

        # Grouped bar chart with train/val/test
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name=f"Train ({train_final:.1f}%)",
                x=display_classes,
                y=train_counts,
                marker_color="#98c127",
                text=train_counts,
                textposition="outside",
            )
        )

        fig.add_trace(
            go.Bar(
                name=f"Validation ({val_final:.1f}%)",
                x=display_classes,
                y=val_counts,
                marker_color="#8fd7d7",
                text=val_counts,
                textposition="outside",
            )
        )

        fig.add_trace(
            go.Bar(
                name=f"Test ({test_final:.1f}%)",
                x=display_classes,
                y=test_counts,
                marker_color="#ffb255",
                text=test_counts,
                textposition="outside",
            )
        )

        title_text = "Samples per Malware Family - Train/Val/Test Split"

    fig.update_layout(
        title=title_text,
        xaxis_title="Malware Family",
        yaxis_title="Number of Samples",
        barmode="stack",
        height=500,
        xaxis={"tickangle": -45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa"),
        showlegend=True,
    )

    st.plotly_chart(fig, width="stretch")

    # Show imbalance status (depends on selected strategy)
    if selected and len(selected) > 1:
        selected_counts = [dataset_info["samples"].get(c, 0) for c in selected]
        original_ratio = max(selected_counts) / max(min(selected_counts), 1)
        imbalance_strategy = st.session_state.get(
            "imbalance_strategy", "Auto Class Weights (Recommended)"
        )

        if imbalance_strategy == "No Adjustment":
            if original_ratio > 10:
                st.error(
                    f"Severe imbalance: {original_ratio:.1f}:1 — no mitigation selected"
                )
            elif original_ratio > 3:
                st.warning(
                    f"Moderate imbalance: {original_ratio:.1f}:1 — consider class weights"
                )
        else:
            st.success(
                f"Imbalance handled. Original {original_ratio:.1f}:1, strategy: _{imbalance_strategy}_"
            )

    # Effective Distribution After Balancing
    render_effective_distribution(display_classes, sample_counts, data_kind)

    # Top and bottom classes (from selected)
    if selected:
        col1, col2 = st.columns(2)

        selected_items = [(c, dataset_info["samples"].get(c, 0)) for c in selected]

        with col1:
            st.markdown("**Most Common Selected Classes**")
            top_5 = sorted(selected_items, key=lambda x: x[1], reverse=True)[:5]
            for cls, count in top_5:
                st.text(f"{cls}: {count:,} samples")

        with col2:
            st.markdown("**Least Common Selected Classes**")
            bottom_5 = sorted(selected_items, key=lambda x: x[1])[:5]
            for cls, count in bottom_5:
                st.text(f"{cls}: {count:,} samples")


def _effective_counts(imbalance_strategy, display_classes, sample_counts, max_samples):
    """Sample counts each class effectively contributes under a strategy"""
    if "Auto Class Weights" in imbalance_strategy:
        return (
            [max_samples] * len(display_classes),
            "All classes contribute **equally** to the training loss",
        )

    if imbalance_strategy == "Selective Augmentation (H2)":
        threshold = st.session_state.get("minority_threshold", 200)
        multiplier = st.session_state.get("aug_multiplier", 2.0)
        return (
            [
                int(count * multiplier) if count < threshold else count
                for count in sample_counts
            ],
            f"Minority classes (<{threshold} samples) get **{multiplier}x** augmentation",
        )

    if imbalance_strategy == "Manual Class Weights":
        weights = st.session_state.get("class_weights", {})
        return (
            [
                int(sample_counts[i] * weights.get(name, 1.0))
                for i, name in enumerate(display_classes)
            ],
            "Effective contribution = samples x weight",
        )

    if imbalance_strategy == "Oversampling (SMOTE)":
        ratio = st.session_state.get("smote_ratio", 0.5)
        target = int(max_samples * ratio)
        smallest_share = min(sample_counts) / max_samples if max_samples else 0
        note = (
            f"SMOTE raises every class to {ratio * 100:.0f}% of the largest "
            f"({target:,} samples), in the training split only"
        )
        if ratio <= smallest_share:
            note += (
                f". The smallest class is already at {smallest_share * 100:.0f}% "
                "of the largest, so raise the ratio above that to see an effect"
            )
        return [max(count, target) for count in sample_counts], note

    if imbalance_strategy == "Undersampling":
        smallest = min(sample_counts) if sample_counts else 1
        return (
            [smallest] * len(display_classes),
            "Every class is cut down to the smallest class size, in the "
            "training split only",
        )

    return sample_counts, ""


def render_effective_distribution(display_classes, sample_counts, data_kind="image_folder"):
    """Show what training will actually see after the imbalance strategy.

    Resampling (SMOTE, undersampling) rewrites the training rows only, and only
    for tabular data - interpolating between image tensors produces nonsense, so
    the image provider does not apply it.
    """
    st.divider()
    st.markdown("### Effective Distribution (After Balancing)")

    imbalance_strategy = st.session_state.get(
        "imbalance_strategy", "Auto Class Weights (Recommended)"
    )

    if imbalance_strategy == "No Adjustment":
        st.info("No balancing applied — see the original distribution above")
        return

    if not sample_counts or not display_classes:
        return

    max_samples = max(sample_counts) if sample_counts else 1

    resampling = imbalance_strategy in ("Oversampling (SMOTE)", "Undersampling")

    # SMOTE interpolates between feature vectors; for images that would blend
    # pixels into samples that mean nothing, so the image provider ignores it.
    if resampling and data_kind != "tabular":
        st.warning(
            f"{imbalance_strategy} applies to tabular data only and is not "
            "applied to image datasets. Use class weights instead."
        )
        return

    effective_counts, explanation = _effective_counts(
        imbalance_strategy, display_classes, sample_counts, max_samples
    )

    if effective_counts == sample_counts:
        st.info(
            f"{explanation}\n\nWith the current settings this leaves the "
            "distribution unchanged."
            if explanation
            else "This strategy leaves the distribution unchanged."
        )
    else:
        st.info(explanation)


    # Build chart with Train/Val/Test split (same format as first chart)
    use_cross_validation = st.session_state.get("use_cross_validation", False)

    fig = go.Figure()

    if use_cross_validation:
        # Cross-validation: single bar per class
        n_folds = st.session_state.get("n_folds", 5)
        fig.add_trace(
            go.Bar(
                name="Effective Samples (CV)",
                x=display_classes,
                y=effective_counts,
                marker_color="#98c127",
                text=effective_counts,
                textposition="outside",
            )
        )
        title_text = f"Effective Distribution - {n_folds}-Fold CV"
    else:
        # Fixed split: Train/Val/Test breakdown
        from utils.dataset_utils import calculate_split_percentages

        train_pct = st.session_state.get("train_split", 70)
        val_of_remaining = st.session_state.get("val_split", 50)
        train_final, val_final, test_final = calculate_split_percentages(
            train_pct, val_of_remaining
        )

        train_eff = [int(c * train_final / 100) for c in effective_counts]
        val_eff = [int(c * val_final / 100) for c in effective_counts]
        test_eff = [int(c * test_final / 100) for c in effective_counts]

        fig.add_trace(
            go.Bar(
                name=f"Train ({train_final:.1f}%)",
                x=display_classes,
                y=train_eff,
                marker_color="#98c127",
                text=train_eff,
                textposition="outside",
            )
        )

        fig.add_trace(
            go.Bar(
                name=f"Validation ({val_final:.1f}%)",
                x=display_classes,
                y=val_eff,
                marker_color="#8fd7d7",
                text=val_eff,
                textposition="outside",
            )
        )

        fig.add_trace(
            go.Bar(
                name=f"Test ({test_final:.1f}%)",
                x=display_classes,
                y=test_eff,
                marker_color="#ffb255",
                text=test_eff,
                textposition="outside",
            )
        )

        title_text = "Effective Distribution - Train/Val/Test Split"

    fig.update_layout(
        title=title_text,
        xaxis_title="Malware Family",
        yaxis_title="Effective Samples",
        barmode="stack",
        height=500,
        xaxis={"tickangle": -45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa"),
        showlegend=True,
    )

    st.plotly_chart(fig, width="stretch")

    # Show improvement metrics
    if effective_counts and sample_counts:
        original_ratio = max(sample_counts) / max(min(sample_counts), 1)
        effective_ratio = max(effective_counts) / max(min(effective_counts), 1)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Ratio", f"{original_ratio:.1f}:1")
        with col2:
            st.metric("Effective Ratio", f"{effective_ratio:.1f}:1")
        with col3:
            improvement = ((original_ratio - effective_ratio) / original_ratio) * 100
            st.metric("Imbalance Reduction", f"{improvement:.0f}%")

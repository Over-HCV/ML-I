"""
Dataset Tab 1: Overview & Split Configuration
Shows dataset statistics and train/val/test split configuration
"""

import plotly.graph_objects as go
import streamlit as st

from content.dataset.tooltips import CONTROL_TOOLTIPS, SECTION_TOOLTIPS
from utils.dataset_registry import DatasetEntry
from utils.dataset_utils import calculate_split_percentages


def render(dataset_info, entry: DatasetEntry):
    """Render overview and split configuration tab"""
    _init_overview_defaults()
    render_dataset_overview(dataset_info, entry)
    st.divider()
    render_data_split(dataset_info)
    st.divider()
    render_class_imbalance_handling(dataset_info)


def _init_overview_defaults():
    """Initialize overview/split defaults if not in session state.

    This must run BEFORE widgets render so widgets read from session_state.
    """
    defaults = {
        "use_cross_validation": False,
        "n_folds": 5,
        "stratified_kfold": True,
        "train_split": 70,
        "val_split": 50,
        "stratified_split": True,
        "random_seed": 73,
        "imbalance_strategy": "Auto Class Weights (Recommended)",
        "minority_threshold": 200,
        "aug_multiplier": 2.0,
        "smote_ratio": 0.5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_dataset_overview(dataset_info, entry: DatasetEntry):
    """Dataset statistics and info"""
    st.subheader("Dataset Overview", help=SECTION_TOOLTIPS["dataset_overview"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", f"{dataset_info['total_samples']:,}")
    with col2:
        st.metric("Number of Classes", len(dataset_info["classes"]))

    st.info(f"Dataset location: `{_display_path(entry)}`")

    # Calculate class imbalance
    if dataset_info["samples"]:
        samples_per_class = list(dataset_info["samples"].values())
        max_samples = max(samples_per_class)
        min_samples = min(samples_per_class)
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Samples/Class", max_samples)
        with col2:
            st.metric("Min Samples/Class", min_samples)

        if imbalance_ratio > 2:
            st.warning(f"Class imbalance detected: {imbalance_ratio:.1f}x ratio")
        else:
            st.success("Classes are relatively balanced")


def _display_path(entry: DatasetEntry) -> str:
    """Path shown to the user: relative to the working directory when possible.

    relative_to() raises when the app is launched from outside the repo, so fall
    back to the absolute path instead of breaking the page.
    """
    from pathlib import Path

    try:
        return entry["path"].relative_to(Path.cwd()).as_posix()
    except ValueError:
        return entry["path"].as_posix()


def render_data_split(dataset_info):
    """Train/val/test split configuration"""
    st.subheader("Train/Validation/Test Split", help=SECTION_TOOLTIPS["data_split"])

    # Current split info
    total = dataset_info["total_samples"]

    if total > 0:
        st.info(f"Total samples available for split: {total}")

    st.markdown("**Configure Split Method:**")

    # Split method selection
    use_cross_validation = st.checkbox(
        "Use Cross-Validation",
        key="use_cross_validation",
        help="Use K-Fold Cross-Validation instead of fixed train/val/test split",
    )

    if use_cross_validation:
        # Cross-Validation configuration
        st.markdown("**K-Fold Cross-Validation Configuration:**")

        col1, col2 = st.columns(2)
        with col1:
            n_folds = st.number_input(
                "Number of Folds (K)",
                min_value=2,
                max_value=20,
                key="n_folds",
                help="Number of folds for cross-validation (typically 5 or 10)",
            )
        with col2:
            st.checkbox(
                "Stratified K-Fold",
                key="stratified_kfold",
                help="Maintain class proportions in each fold",
            )

        # Show fold distribution
        fold_pct = 100 / n_folds
        train_fold_pct = ((n_folds - 1) / n_folds) * 100
        val_fold_pct = (1 / n_folds) * 100

        st.info(f"""
        **Cross-Validation Setup:**
        - Each fold: {fold_pct:.1f}% of data
        - Training per iteration: {train_fold_pct:.1f}% ({n_folds - 1} folds)
        - Validation per iteration: {val_fold_pct:.1f}% (1 fold)
        - Total iterations: {n_folds}
        """)

        # Visualization for CV
        fig = go.Figure()
        for i in range(n_folds):
            fig.add_trace(
                go.Bar(
                    name=f"Fold {i + 1}",
                    x=["Training", "Validation"],
                    y=[train_fold_pct if i != 0 else 0, val_fold_pct if i == 0 else 0],
                    marker_color="#98c127" if i == 0 else "#8fd7d7",
                )
            )

        fig.update_layout(
            title=f"{n_folds}-Fold Cross-Validation Distribution",
            xaxis_title="Split Type",
            yaxis_title="Percentage (%)",
            barmode="stack",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        st.number_input(
            "Random Seed",
            min_value=0,
            key="random_seed",
            help="For reproducible fold splits",
        )
    else:
        # Fixed train/val/test split
        st.markdown("**Fixed Train/Validation/Test Split:**")

        # Slider 1: Training percentage
        train_pct = st.slider(
            "Training %",
            min_value=0,
            max_value=100,
            step=5,
            key="train_split",
            help="Percentage of data for training",
        )

        # Slider 2: Validation percentage from remaining
        remaining = 100 - train_pct
        val_of_remaining = st.slider(
            f"Validation % (of remaining {remaining}%)",
            min_value=0,
            max_value=100,
            step=5,
            key="val_split",
            help="Percentage of remaining data for validation (rest goes to test)",
        )

        # Calculate final percentages
        train_final, val_final, test_final = calculate_split_percentages(
            train_pct, val_of_remaining
        )

        # Display final split
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train", f"{train_final:.1f}%")
        with col2:
            st.metric("Validation", f"{val_final:.1f}%")
        with col3:
            st.metric("Test", f"{test_final:.1f}%")

        # The metrics above already give the percentages, so the chart carries
        # the sample counts instead of repeating them.
        train_count = int(total * train_final / 100)
        val_count = int(total * val_final / 100)
        test_count = total - train_count - val_count

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Train", "Validation", "Test"],
                    values=[train_count, val_count, test_count],
                    marker_colors=["#98c127", "#8fd7d7", "#ffb255"],
                    hole=0.3,
                    text=[f"{count:,}" for count in (train_count, val_count, test_count)],
                    textinfo="label+text",
                    hovertemplate=(
                        "%{label}<br>%{value:,} samples<br>"
                        "%{percent} of {total:,}<extra></extra>"
                    ).replace("{total:,}", f"{total:,}"),
                )
            ]
        )
        fig.update_layout(
            title=f"Samples per Split (total {total:,})",
            height=300,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.checkbox(
                "Stratified Split",
                key="stratified_split",
                help="Maintain class proportions in each split",
            )
        with col2:
            st.number_input(
                "Random Seed",
                min_value=0,
                key="random_seed",
                help="For reproducible splits",
            )


def render_class_imbalance_handling(dataset_info):
    """Class imbalance mitigation options"""
    st.subheader("Class Imbalance Handling", help=SECTION_TOOLTIPS["class_imbalance"])

    # Calculate imbalance for selected classes
    if "selected_classes" in st.session_state and st.session_state.selected_classes:
        selected_classes = st.session_state.selected_classes
        samples_per_class = [
            dataset_info["samples"].get(c, 0) for c in selected_classes
        ]
    else:
        samples_per_class = list(dataset_info["samples"].values())

    if samples_per_class:
        max_samples = max(samples_per_class)
        min_samples = min(samples_per_class) if min(samples_per_class) > 0 else 1
        imbalance_ratio = max_samples / min_samples

        # Show imbalance severity
        if imbalance_ratio > 10:
            st.error(
                f"Severe imbalance: {imbalance_ratio:.1f}:1 between the largest and smallest class"
            )
        elif imbalance_ratio > 3:
            st.warning(f"Moderate imbalance: {imbalance_ratio:.1f}:1 ratio")
        else:
            st.success(f"Balanced: {imbalance_ratio:.1f}:1 ratio")

    # Handling strategy selection
    st.markdown("### Mitigation Strategy")

    strategy = st.radio(
        "Select imbalance handling method",
        [
            "Auto Class Weights (Recommended)",
            "Selective Augmentation (H2)",
            "Manual Class Weights",
            "Oversampling (SMOTE)",
            "Undersampling",
            "No Adjustment",
        ],
        key="imbalance_strategy",
        help="Choose how to handle class imbalance during training",
    )

    # Strategy-specific options
    if strategy == "Auto Class Weights (Recommended)":
        st.info("""
        **Auto Class Weights**: Automatically calculates class weights inversely proportional to class frequencies.
        - Classes with fewer samples get higher weights
        - Balanced loss function during training
        - No data duplication or removal
        """)

    elif strategy == "Selective Augmentation (H2)":
        st.info("""
        **Selective Augmentation** (supports H2 hypothesis):
        - Applies MORE augmentation to minority classes only
        - Majority classes receive standard augmentation
        - Tests if augmentation improves minority class recall
        """)

        col1, col2 = st.columns(2)
        with col1:
            minority_threshold = st.number_input(
                "Minority threshold (samples)",
                min_value=50,
                max_value=500,
                step=50,
                key="minority_threshold",
                help="Classes with fewer samples than this are considered 'minority'",
            )
        with col2:
            aug_multiplier = st.slider(
                "Augmentation multiplier",
                min_value=1.5,
                max_value=5.0,
                step=0.5,
                key="aug_multiplier",
                help="How much MORE augmentation minority classes receive",
            )

        # Show which classes qualify as minority
        if "selected_classes" in st.session_state and st.session_state.selected_classes:
            minority_classes = [
                c
                for c in st.session_state.selected_classes
                if dataset_info["samples"].get(c, 0) < minority_threshold
            ]
            majority_classes = [
                c
                for c in st.session_state.selected_classes
                if dataset_info["samples"].get(c, 0) >= minority_threshold
            ]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Minority Classes", len(minority_classes))
                if minority_classes:
                    with st.expander("View minority classes"):
                        for c in sorted(minority_classes):
                            st.text(
                                f"• {c}: {dataset_info['samples'].get(c, 0)} samples"
                            )
            with col2:
                st.metric("Majority Classes", len(majority_classes))

            # Store in session state
            st.session_state.minority_classes = minority_classes
            st.session_state.majority_classes = majority_classes

    elif strategy == "Manual Class Weights":
        st.markdown("**Set custom weights for each class:**")

        # Get selected classes
        if "selected_classes" in st.session_state and st.session_state.selected_classes:
            classes_to_weight = sorted(st.session_state.selected_classes)
        else:
            classes_to_weight = sorted(dataset_info["classes"])

        # Create weight inputs for each class
        weights = {}
        cols = st.columns(3)
        for i, cls in enumerate(classes_to_weight):
            col_idx = i % 3
            with cols[col_idx]:
                count = dataset_info["samples"].get(cls, 0)
                default_weight = 1.0 if count == 0 else (max_samples / count)
                # Initialize weight in session state if not present
                weight_key = f"weight_{cls}"
                if weight_key not in st.session_state:
                    st.session_state[weight_key] = min(default_weight, 10.0)

                weights[cls] = st.number_input(
                    f"{cls} ({count} samples)",
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    key=weight_key,
                    help=f"Weight for {cls} class",
                )

        # Store weights in session state
        st.session_state.class_weights = weights

    elif strategy == "Oversampling (SMOTE)":
        st.warning("""
        ⚠️ **Warning: SMOTE is NOT recommended for image data!**

        SMOTE interpolates between feature vectors, which creates unrealistic
        synthetic images (blended pixels). Consider using:
        - **Auto Class Weights** (recommended) - adjusts loss function
        - **Selective Augmentation** - more augmentation for minority classes
        """)

        sampling_ratio = st.slider(
            "Sampling Ratio",
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            key="smote_ratio",
            help="Target ratio of minority to majority class after resampling",
        )

    elif strategy == "Undersampling":
        st.info("""
        **Random Undersampling**:
        - Reduces samples from majority classes
        - Balances dataset by removing data
        - May lose important information
        """)

        st.warning("Undersampling reduces the training set size")

    elif strategy == "No Adjustment":
        st.info("""
        **No Adjustment**:
        - Train with natural class distribution
        - May lead to bias towards majority classes
        - Consider if imbalance reflects real-world distribution
        """)

    # Store strategy in session state
    st.session_state.imbalance_handling = strategy

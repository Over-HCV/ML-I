"""
Dataset Tab (tabular): Columns & Preprocessing
Feature selection, encoding options, and configuration save
"""

import pandas as pd
import streamlit as st

from state.workflow import has_dataset_config, save_dataset_config
from utils.dataset_registry import DatasetEntry
from utils.dataset_utils import calculate_split_percentages
from utils.tabular_utils import (
    ENCODING_OPTIONS,
    MISSING_OPTIONS,
    SCALING_OPTIONS,
    feature_matrix_width,
    load_dataframe,
    split_column_types,
)

PREVIEW_ROWS = 10


def render(dataset_info, entry: DatasetEntry, target_column: str):
    """Render feature configuration and the configuration save"""
    try:
        frame = load_dataframe(entry["path"])
    except Exception as exception:
        st.error(f"Could not read {entry['name']}: {exception}")
        return

    _init_defaults(frame, target_column)

    feature_columns = _render_feature_selection(frame, target_column)
    st.divider()
    preprocessing = _render_preprocessing_options(frame, feature_columns)
    st.divider()
    _render_configuration_summary(
        dataset_info, entry, target_column, feature_columns, preprocessing
    )


def _init_defaults(frame: pd.DataFrame, target_column: str) -> None:
    """Seed widget defaults before the widgets render"""
    if "tabular_features" not in st.session_state:
        st.session_state.tabular_features = [
            column for column in frame.columns if column != target_column
        ]

    defaults = {
        "tabular_scaling": "Standard",
        "tabular_encoding": "One-hot",
        "tabular_missing": MISSING_OPTIONS[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_feature_selection(frame: pd.DataFrame, target_column: str) -> list[str]:
    """Choose which columns are inputs to the model"""
    st.subheader("Feature Columns")

    available = [column for column in frame.columns if column != target_column]

    # The target must never appear as a feature; a stored selection from another
    # target column can still contain it.
    current = [
        column
        for column in st.session_state.get("tabular_features", available)
        if column in available
    ]
    st.session_state.tabular_features = current or available

    col1, _ = st.columns([1, 3])
    with col1:
        # Keyed because the Class Distribution tab has an identical button
        if st.button("Select All", key="select_all_features", width="stretch"):
            st.session_state.tabular_features = available
            st.rerun()

    selected = st.multiselect(
        "Columns used as model input",
        options=available,
        default=st.session_state.tabular_features,
        key="feature_selector",
        help=f"Everything except the target column '{target_column}'",
    )
    st.session_state.tabular_features = selected

    if not selected:
        st.warning("No feature columns selected. Select at least one column.")
        return []

    numeric, categorical = split_column_types(frame, selected)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected", len(selected))
    with col2:
        st.metric("Numeric", len(numeric))
    with col3:
        st.metric("Categorical", len(categorical))

    with st.expander(f"Preview first {PREVIEW_ROWS} rows"):
        st.dataframe(
            frame[selected + [target_column]].head(PREVIEW_ROWS), width="stretch"
        )

    return selected


def _render_preprocessing_options(
    frame: pd.DataFrame, feature_columns: list[str]
) -> dict:
    """Scaling, encoding and missing-value handling"""
    st.subheader("Preprocessing")

    col1, col2, col3 = st.columns(3)

    with col1:
        scaling = st.selectbox(
            "Numeric scaling",
            SCALING_OPTIONS,
            key="tabular_scaling",
            help="Standard centres each column; MinMax squeezes it into [0, 1]",
        )
    with col2:
        encoding = st.selectbox(
            "Categorical encoding",
            ENCODING_OPTIONS,
            key="tabular_encoding",
            help="One-hot adds a column per category; Ordinal maps them to integers",
        )
    with col3:
        missing = st.selectbox(
            "Missing values",
            MISSING_OPTIONS,
            key="tabular_missing",
            help="How to handle empty cells in the feature columns",
        )

    preprocessing = {
        "numeric_scaling": scaling,
        "categorical_encoding": encoding,
        "missing_values": missing,
    }

    if not feature_columns:
        return preprocessing

    width = feature_matrix_width(frame, feature_columns, scaling, encoding)
    preprocessing["input_dim"] = width

    st.info(
        f"Feature matrix: {len(frame):,} rows x {width} encoded columns. "
        "This width is the model's input size."
    )

    st.caption(
        "Scaler and imputer statistics are fitted on the training split only, so "
        "the validation and test rows stay unseen."
    )

    return preprocessing


def _render_configuration_summary(
    dataset_info,
    entry: DatasetEntry,
    target_column: str,
    feature_columns: list[str],
    preprocessing: dict,
) -> None:
    """Build, show and save the dataset configuration"""
    st.subheader("Configuration")

    selected_classes = st.session_state.get(
        "selected_classes", dataset_info["classes"]
    )

    if not selected_classes:
        st.warning(
            "No classes are selected, so there is nothing to configure. Pick at "
            "least one class on the **Class Distribution** tab."
        )
        return

    total_samples = sum(
        dataset_info["samples"].get(name, 0) for name in selected_classes
    )

    config = _build_config(
        entry, target_column, feature_columns, preprocessing, selected_classes,
        total_samples,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classes", len(selected_classes))
    with col2:
        st.metric("Rows", f"{total_samples:,}")
    with col3:
        st.metric("Features", preprocessing.get("input_dim", 0))

    can_save = bool(feature_columns) and bool(selected_classes)

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if st.button(
            "Save Configuration",
            type="primary",
            width="stretch",
            disabled=not can_save,
        ):
            save_dataset_config(config)
            st.rerun()

    if has_dataset_config():
        st.success("Configuration saved. Continue on the Model page.")

    with st.expander("View Full Configuration", expanded=False):
        st.json(config)


def _build_config(
    entry: DatasetEntry,
    target_column: str,
    feature_columns: list[str],
    preprocessing: dict,
    selected_classes: list[str],
    total_samples: int,
) -> dict:
    """Assemble the dataset config in the same shape the image path writes"""
    if st.session_state.get("use_cross_validation", False):
        split_config = {
            "method": "cross_validation",
            "n_folds": st.session_state.get("n_folds", 5),
            "stratified": st.session_state.get("stratified_kfold", True),
            "random_seed": st.session_state.get("random_seed", 73),
        }
    else:
        train_final, val_final, test_final = calculate_split_percentages(
            st.session_state.get("train_split", 70),
            st.session_state.get("val_split", 50),
        )
        split_config = {
            "method": "fixed_split",
            "train": round(train_final, 1),
            "val": round(val_final, 1),
            "test": round(test_final, 1),
            "stratified": st.session_state.get("stratified_split", True),
            "random_seed": st.session_state.get("random_seed", 73),
        }

    imbalance_strategy = st.session_state.get(
        "imbalance_strategy", "Auto Class Weights (Recommended)"
    )

    return {
        "dataset_kind": "tabular",
        "task": "classification",
        "dataset_name": entry["name"],
        "dataset_path": str(entry["path"]),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "input_dim": preprocessing.get("input_dim", 0),
        "total_samples": total_samples,
        "num_classes": len(selected_classes),
        # Named selected_families to match the image config: the Model page
        # derives num_classes from this key for either kind of dataset.
        "selected_families": selected_classes,
        "split": split_config,
        "preprocessing": preprocessing,
        "imbalance_handling": {
            "strategy": imbalance_strategy,
            "class_weights": st.session_state.get("class_weights")
            if imbalance_strategy == "Manual Class Weights"
            else None,
        },
    }

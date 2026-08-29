"""
Dataset selector - pick the active dataset, or add one from a URL
"""

import pandas as pd
import streamlit as st

from state.cache import clear_dataset_info
from state.workflow import get_active_dataset, get_dataset_config, set_active_dataset
from utils.dataset_download import (
    MAX_DOWNLOAD_BYTES,
    DownloadError,
    download_csv,
    filename_from_url,
)
from utils.dataset_registry import (
    DatasetEntry,
    default_dataset_name,
    list_datasets,
)
from utils.tabular_utils import REGRESSION, infer_task, load_dataframe

CSV_PREVIEW_ROWS = 20
MAX_DOWNLOAD_MB = MAX_DOWNLOAD_BYTES // (1024 * 1024)

# A download ends in st.rerun(), which clears anything already written to the
# page, so the confirmation has to survive one rerun in session state.
_DOWNLOAD_FLASH_KEY = "_dataset_download_flash"

_LAYOUT_HELP = """
Datasets live in `learn/app/data`. Add them from the editor and press Rescan.

An image dataset is one folder per dataset, one subfolder per class:

```
learn/app/data/
  my-dataset/
    ClassA/image_001.png
    ClassB/image_002.png
```

Image types read: `.png`, `.jpg`, `.jpeg`, `.bmp`. A folder without a class
subfolder holding images is skipped.

A tabular dataset is a single CSV at the top level:

```
learn/app/data/
  measurements.csv
```
"""


def render_dataset_selector() -> DatasetEntry | None:
    """Render the dataset picker.

    Returns the selected entry, or None when there is nothing to work with -
    the caller should stop rendering in that case.
    """
    datasets = list_datasets()

    if not datasets:
        _render_empty_state()
        return None

    names = [entry["name"] for entry in datasets]
    active_name = _resolve_active_name(names, datasets)

    col_picker, col_rescan = st.columns([4, 1])

    with col_picker:
        selected_name = st.selectbox(
            "Active dataset",
            names,
            index=names.index(active_name),
            help="Everything found in learn/app/data",
        )

    with col_rescan:
        # Vertical padding so the button lines up with the selectbox
        st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
        if st.button("Rescan", width="stretch", help="Re-read learn/app/data"):
            clear_dataset_info()
            st.rerun()

    if selected_name != active_name:
        set_active_dataset(selected_name)
        st.rerun()

    entry = next(item for item in datasets if item["name"] == selected_name)
    _render_entry_summary(entry)

    with st.expander("Add a dataset"):
        _render_url_form()
        st.divider()
        st.markdown(_LAYOUT_HELP)

    return entry


def _resolve_active_name(names: list[str], datasets: list[DatasetEntry]) -> str:
    """Return the active dataset name, recovering if it no longer exists.

    A saved session can name a dataset that has since been renamed or deleted.
    """
    active_name = get_active_dataset()
    if active_name in names:
        return active_name

    fallback = default_dataset_name(datasets)
    if active_name:
        st.warning(
            f"Dataset '{active_name}' from the saved session is no longer in "
            f"learn/app/data. Showing '{fallback}' instead."
        )

    set_active_dataset(fallback)
    return fallback


def _render_empty_state() -> None:
    """Shown when learn/app/data holds no usable dataset"""
    st.warning("No datasets found.")

    _render_url_form()
    st.divider()
    st.markdown(_LAYOUT_HELP)

    if st.button("Rescan", help="Re-read learn/app/data"):
        clear_dataset_info()
        st.rerun()


def _render_url_form() -> None:
    """Download a CSV dataset from a URL"""
    st.markdown("**From a URL**")

    flash = st.session_state.pop(_DOWNLOAD_FLASH_KEY, "")
    if flash:
        st.success(flash)

    url = st.text_input(
        "CSV URL",
        key="dataset_url",
        placeholder="https://example.com/data.csv",
        help=f"Direct link to a CSV over http or https, up to {MAX_DOWNLOAD_MB} MB",
    )

    col_name, col_replace = st.columns([3, 1])

    with col_name:
        name = st.text_input(
            "Save as",
            key="dataset_url_name",
            placeholder=filename_from_url(url) if url.strip() else "data.csv",
            help="Leave empty to use the filename from the URL",
        )

    with col_replace:
        st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
        overwrite = st.checkbox(
            "Replace",
            key="dataset_url_overwrite",
            help="Overwrite a dataset that already has this name",
        )

    if not st.button("Download", type="primary", disabled=not url.strip()):
        return

    try:
        with st.spinner("Downloading..."):
            path = download_csv(url, name=name, overwrite=overwrite)
    except DownloadError as exception:
        st.error(str(exception))
        return

    size_kb = path.stat().st_size / 1024
    st.session_state[_DOWNLOAD_FLASH_KEY] = f"Saved {path.name} ({size_kb:,.0f} KB)"

    set_active_dataset(path.name)
    st.rerun()


def render_target_selector(entry: DatasetEntry) -> str | None:
    """Choose which column of a CSV holds the label.

    Rendered above the tabs because the split and distribution views are
    meaningless until the target is known. Returns the column name, or None when
    the file cannot be read or the target is unusable.
    """
    try:
        frame = load_dataframe(entry["path"])
    except Exception as exception:
        st.error(f"Could not read {entry['name']}: {exception}")
        return None

    columns = list(frame.columns)
    if not columns:
        st.error(f"{entry['name']} has no columns.")
        return None

    # Restore a saved choice, else fall back to the last column - the label is
    # conventionally last, which is right for most teaching datasets.
    stored = st.session_state.get("csv_target_column") or _saved_target_column(entry)
    default_index = columns.index(stored) if stored in columns else len(columns) - 1

    target_column = st.selectbox(
        "Target column",
        columns,
        index=default_index,
        key="csv_target_column",
        help="The column the model learns to predict",
    )

    if infer_task(frame[target_column]) == REGRESSION:
        st.warning(
            f"'{target_column}' looks like a continuous quantity, not a set of "
            "labels. Regression is not supported yet - pick a column with a small "
            "number of repeated values."
        )
        return None

    return target_column


def _saved_target_column(entry: DatasetEntry) -> str:
    """Target column from the saved config, when it belongs to this dataset"""
    config = get_dataset_config()
    if config.get("dataset_name") != entry["name"]:
        return ""

    return config.get("target_column", "")


def _render_entry_summary(entry: DatasetEntry) -> None:
    """One-line summary of the selected dataset"""
    if entry["kind"] == "csv":
        summary = f"Tabular · {entry['num_samples']:,} rows"
    else:
        summary = (
            f"Images · {entry['num_classes']} classes · {entry['num_samples']:,} files"
        )

    st.caption(summary)

    if entry["source_url"]:
        st.caption(f"Source: {entry['source_url']}")


def render_csv_preview(entry: DatasetEntry) -> None:
    """Preview a CSV dataset.

    Tabular datasets are browsable only - the model, training and
    interpretability pages all assume image folders.
    """
    st.subheader(entry["name"])
    st.info(
        "Tabular training is not supported yet. The Model and Training pages "
        "need an image dataset."
    )

    try:
        dataframe = pd.read_csv(entry["path"], nrows=CSV_PREVIEW_ROWS)
    except Exception as exception:
        st.error(f"Could not read {entry['name']}: {exception}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", f"{entry['num_samples']:,}")
    with col2:
        st.metric("Columns", len(dataframe.columns))

    st.markdown(f"**First {min(CSV_PREVIEW_ROWS, len(dataframe))} rows**")
    st.dataframe(dataframe, width="stretch")

    st.markdown("**Columns**")
    schema = pd.DataFrame(
        {
            "column": dataframe.columns,
            "dtype": [str(dtype) for dtype in dataframe.dtypes],
            "non_null_in_preview": [
                int(dataframe[column].notna().sum()) for column in dataframe.columns
            ],
        }
    )
    st.dataframe(schema, width="stretch", hide_index=True)

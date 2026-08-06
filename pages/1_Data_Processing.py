import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import scipy.io
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import matplotlib.pyplot as plt

# =========================================================
# PAGE BRANDING
# =========================================================
st.markdown("""
    <h2 style='text-align: center; color: #B31B1B; margin-bottom: 0.1rem;'>
        Tab 1 - Data Loading, Feature Engineering, and Visualization
    </h2>
    <p style='text-align: center; color: gray; margin-top: 0;'>
        iCAAV Core - Advanced Biomechatronics and Locomotion Laboratory - Carleton University
    </p>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
defaults = {
    "df": None,                # raw / preprocessed dataset
    "label_col": None,         # user-designated label/class column
    "feature_cols": [],        # user-designated feature/variable columns
    "engineered_df": None,     # extracted statistical feature set
    "saved_feature_sets": {},  # in-session named feature sets
    "selected_important_df": None,  # post-visualization feature subset for export
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_pickle_bytes(obj) -> bytes:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.getvalue()


def to_npy_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    np.save(buf, df.select_dtypes(include=[np.number]).values)
    return buf.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def download_fig_button(fig, key, filename):
    st.download_button(
        "Download This Plot (PNG)",
        fig_to_png_bytes(fig),
        file_name=filename,
        mime="image/png",
        key=key,
    )


# Metadata-like columns can dominate scaling and PCA even though they do not
# represent measured driving behavior, so the exclusion logic is centralized.
def is_metadata_column(col_name: str) -> bool:
    name = str(col_name).strip().lower()
    exact_names = {
        "id", "index", "idx", "start_idx", "end_idx", "row", "row_id",
        "time", "timestamp", "date", "datetime",
    }
    if name in exact_names:
        return True
    return (
        name.endswith("_id")
        or name.endswith("_idx")
        or "timestamp" in name
        or name.startswith("time_")
        or name.endswith("_time")
    )


# Most analysis sections need numeric signal features, not labels or metadata;
# this helper keeps those choices consistent across preprocessing and PCA.
def numeric_feature_columns(
    df: pd.DataFrame,
    label_col: str | None = None,
    exclude_metadata: bool = True,
) -> list[str]:
    if df is None or df.empty:
        return []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = set()
    if label_col in df.columns:
        excluded.add(label_col)
    if exclude_metadata:
        excluded.update(c for c in numeric_cols if is_metadata_column(c))
    return [c for c in numeric_cols if c not in excluded]


# Constant and near-constant fields add no meaningful PCA variance and can make
# feature summaries look healthier than the data really is.
def constant_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].nunique(dropna=False) <= 1]


def near_constant_columns(
    df: pd.DataFrame,
    cols: list[str],
    threshold: float = 0.995,
) -> list[str]:
    near_constant = []
    for c in cols:
        if c not in df.columns:
            continue
        shares = df[c].value_counts(dropna=False, normalize=True)
        if len(shares) > 1 and shares.iloc[0] >= threshold:
            near_constant.append(c)
    return near_constant


# PCA and feature importance need a reliable target for coloring/scoring; very
# high-cardinality numeric columns are treated as continuous measurements.
def is_label_like(series: pd.Series) -> bool:
    if series is None or series.empty:
        return False
    unique_count = series.nunique(dropna=True)
    return 1 < unique_count <= 20 or not pd.api.types.is_numeric_dtype(series)


def safe_mode(series: pd.Series):
    mode_vals = series.dropna().mode()
    return mode_vals.iloc[0] if not mode_vals.empty else None


def missing_value_counts(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False)


# Mode imputation is useful for categorical fields and discrete class-like values,
# so it fills each column with its own most frequent non-missing value.
def fill_missing_with_mode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    filled_df = df.copy()
    skipped_cols = []
    for col in filled_df.columns:
        mode_values = filled_df[col].dropna().mode()
        if mode_values.empty:
            skipped_cols.append(col)
            continue
        filled_df[col] = filled_df[col].fillna(mode_values.iloc[0])
    return filled_df, skipped_cols


# Shared visualization controls are converted into one config dictionary so
# every Matplotlib plot reads the same size, scale, bins, and zoom settings.
def build_plot_config(
    width: int,
    height: int,
    zoom_percent: int,
    axis_scale: str,
    hist_bins: int,
) -> dict:
    return {
        "width": width,
        "height": height,
        "figsize": (width, height),
        "zoom_percent": zoom_percent,
        "axis_scale": axis_scale,
        "hist_bins": hist_bins,
    }


def finite_numeric_values(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    numeric = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    return numeric.to_numpy(dtype=float)


def padded_axis_limits(values, zoom_percent: int, log_scale: bool = False):
    vals = finite_numeric_values(values)
    if log_scale:
        vals = vals[vals > 0]
    if vals.size == 0:
        return None

    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    zoom_factor = zoom_percent / 100

    if np.isclose(v_min, v_max):
        pad = max(abs(v_min) * 0.05, 1.0)
        if log_scale:
            v_min = max(v_min / 1.5, np.nextafter(0, 1))
            v_max = v_max * 1.5
        else:
            v_min -= pad
            v_max += pad

    if log_scale:
        log_min = np.log10(v_min)
        log_max = np.log10(v_max)
        center = (log_min + log_max) / 2
        half_range = max((log_max - log_min) * 1.05 / 2, 0.05) * zoom_factor
        return 10 ** (center - half_range), 10 ** (center + half_range)

    center = (v_min + v_max) / 2
    half_range = max((v_max - v_min) * 1.05 / 2, 0.5) * zoom_factor
    return center - half_range, center + half_range


def set_axis_scale_safely(ax, axis: str, axis_scale: str, values) -> str:
    vals = finite_numeric_values(values)
    if axis_scale == "Log":
        if vals.size > 0 and np.all(vals > 0):
            getattr(ax, f"set_{axis}scale")("log")
            return "Log"
        getattr(ax, f"set_{axis}scale")("linear")
        return "Linear"
    if axis_scale == "Symmetric log":
        getattr(ax, f"set_{axis}scale")("symlog")
        return "Symmetric log"
    getattr(ax, f"set_{axis}scale")("linear")
    return "Linear"


# Below 100% zooms in and above 100% zooms out by resizing limits around the
# plotted data range, not the browser zoom or Streamlit page scale.
def apply_plot_controls(
    ax,
    config: dict,
    x_values=None,
    y_values=None,
    apply_x: bool = True,
    apply_y: bool = True,
):
    if apply_x:
        x_scale = set_axis_scale_safely(ax, "x", config["axis_scale"], x_values)
        x_limits = padded_axis_limits(x_values, config["zoom_percent"], log_scale=x_scale == "Log")
        if x_limits is not None:
            ax.set_xlim(*x_limits)
    if apply_y:
        y_scale = set_axis_scale_safely(ax, "y", config["axis_scale"], y_values)
        y_limits = padded_axis_limits(y_values, config["zoom_percent"], log_scale=y_scale == "Log")
        if y_limits is not None:
            ax.set_ylim(*y_limits)


# Matplotlib 3D axes do not support the same safe axis-scale behavior as 2D axes,
# so 3D uses the shared size and zoom only.
def apply_3d_zoom_controls(ax, config: dict, x_values, y_values, z_values):
    x_limits = padded_axis_limits(x_values, config["zoom_percent"])
    y_limits = padded_axis_limits(y_values, config["zoom_percent"])
    z_limits = padded_axis_limits(z_values, config["zoom_percent"])
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if z_limits is not None:
        ax.set_zlim(*z_limits)


# User-selected important features are stored separately so downloads do not
# mutate the working dataset or engineered feature set.
def build_important_feature_subset(
    source_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str | None = None,
    include_label: bool = True,
) -> pd.DataFrame:
    selected_cols = []
    if include_label and label_col in source_df.columns:
        selected_cols.append(label_col)
    selected_cols.extend([c for c in feature_cols if c in source_df.columns])
    selected_cols = list(dict.fromkeys(selected_cols))
    return source_df[selected_cols].copy() if selected_cols else pd.DataFrame(index=source_df.index)


# PCA needs a clean feature matrix and an audit trail of what was removed so the
# user can interpret the result instead of silently fitting on unsuitable fields.
def prepare_pca_feature_matrix(
    df: pd.DataFrame,
    label_col: str | None,
    near_constant_threshold: float = 0.995,
) -> tuple[pd.DataFrame, dict]:
    candidate_cols = numeric_feature_columns(df, label_col, exclude_metadata=True)
    report = {
        "candidate_cols": candidate_cols,
        "constant_cols": [],
        "near_constant_cols": [],
        "all_missing_cols": [],
        "missing_values_before_imputation": 0,
        "rows_dropped_after_imputation": 0,
    }
    if not candidate_cols:
        return pd.DataFrame(index=df.index), report

    feature_df = df[candidate_cols].replace([np.inf, -np.inf], np.nan).copy()
    report["constant_cols"] = constant_columns(feature_df, feature_df.columns.tolist())
    remaining_cols = [c for c in feature_df.columns if c not in report["constant_cols"]]
    report["near_constant_cols"] = near_constant_columns(
        feature_df,
        remaining_cols,
        threshold=near_constant_threshold,
    )
    drop_cols = report["constant_cols"] + report["near_constant_cols"]
    feature_df = feature_df.drop(columns=drop_cols, errors="ignore")

    report["all_missing_cols"] = [
        c for c in feature_df.columns if feature_df[c].notna().sum() == 0
    ]
    feature_df = feature_df.drop(columns=report["all_missing_cols"], errors="ignore")
    report["missing_values_before_imputation"] = int(feature_df.isna().sum().sum())

    if feature_df.empty:
        return feature_df, report

    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    valid_rows = feature_df.notna().all(axis=1)
    report["rows_dropped_after_imputation"] = int((~valid_rows).sum())
    feature_df = feature_df.loc[valid_rows]
    return feature_df, report


# =========================================================
# 3.1.1 DATA IMPORT
# =========================================================
st.header("3.1.1 Data Import")

uploaded_file = st.file_uploader(
    "Import Dataset (CSV, Excel, or MATLAB .mat)",
    type=["csv", "xlsx", "xls", "mat"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            new_df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".mat"):
            mat_dict = scipy.io.loadmat(uploaded_file)
            mat_vars = [k for k in mat_dict.keys() if not k.startswith("__")]
            if not mat_vars:
                st.error("No usable variables found in this .mat file.")
                new_df = None
            else:
                var_choice = st.selectbox("Select MATLAB variable to import", mat_vars)
                array = np.asarray(mat_dict[var_choice])
                if array.ndim == 1:
                    array = array.reshape(-1, 1)
                elif array.ndim > 2:
                    array = array.reshape(array.shape[0], -1)
                new_df = pd.DataFrame(
                    array, columns=[f"col_{i+1}" for i in range(array.shape[1])]
                )
        else:
            new_df = None

        if new_df is not None:
            st.session_state.df = new_df
            st.session_state.feature_cols = []
            st.session_state.label_col = None

    except Exception as e:
        st.error(f"Failed to load file: {e}")

df = st.session_state.df

if df is not None:
    st.success(f"Dataset Loaded - Shape: {df.shape}")
    st.dataframe(df.head())

    all_cols = df.columns.tolist()
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols_all = [c for c in all_cols if c not in numeric_cols_all]

    # The overview gives quick quality checks before the user makes modeling
    # choices, which helps catch data issues while staying before supervised ML.
    st.subheader("Dataset Overview")
    overview_cols = st.columns(4)
    overview_cols[0].metric("Rows", f"{df.shape[0]:,}")
    overview_cols[1].metric("Columns", f"{df.shape[1]:,}")
    overview_cols[2].metric("Numeric", f"{len(numeric_cols_all):,}")
    overview_cols[3].metric("Categorical", f"{len(categorical_cols_all):,}")

    dtype_summary = pd.DataFrame({
        "Column": all_cols,
        "Data Type": [str(df[c].dtype) for c in all_cols],
        "Missing Values": [int(df[c].isna().sum()) for c in all_cols],
        "Unique Values": [int(df[c].nunique(dropna=True)) for c in all_cols],
    })
    with st.expander("Column Data-Type and Quality Summary", expanded=False):
        st.dataframe(dtype_summary)

    missing_overview = missing_value_counts(df)
    missing_overview = missing_overview[missing_overview > 0]
    if missing_overview.empty:
        st.caption("No missing values detected in the loaded dataset.")
    else:
        st.warning("Missing values are present and should be handled before PCA or model training.")
        st.dataframe(missing_overview.rename("missing_count"))

    metadata_cols = [c for c in all_cols if is_metadata_column(c)]
    constant_cols_all = constant_columns(df, numeric_cols_all)
    near_constant_cols_all = near_constant_columns(df, numeric_cols_all)
    if metadata_cols:
        st.caption(f"Metadata-like columns detected: {', '.join(metadata_cols)}")
    if constant_cols_all:
        st.warning(f"Constant numeric columns detected: {', '.join(constant_cols_all)}")
    if near_constant_cols_all:
        st.warning(f"Near-constant numeric columns detected: {', '.join(near_constant_cols_all)}")

    st.subheader("Define Variables, Labels, and Channels")
    c1, c2 = st.columns(2)
    current_label = st.session_state.label_col if st.session_state.label_col in all_cols else None
    default_feature_cols = numeric_feature_columns(df, current_label)

    with c1:
        selected_feature_cols = st.multiselect(
            "Define Variables/Features",
            all_cols,
            default=default_feature_cols,
            help="Columns treated as measured signals/variables."
        )

    with c2:
        label_options = ["None"] + all_cols
        default_idx = 0
        if st.session_state.label_col in all_cols:
            default_idx = label_options.index(st.session_state.label_col)
        label_choice = st.selectbox(
            "Define Label/Class Column (optional)", label_options, index=default_idx,
            help="Column identifying class/label, used for coloring plots and PCA."
        )
        st.session_state.label_col = None if label_choice == "None" else label_choice

    feature_cols = [
        c for c in selected_feature_cols
        if c != st.session_state.label_col and not is_metadata_column(c)
    ]
    removed_features = [c for c in selected_feature_cols if c not in feature_cols]
    if removed_features:
        st.info(
            "Excluded label/metadata column(s) from feature processing: "
            + ", ".join(removed_features)
        )
    st.session_state.feature_cols = feature_cols

    non_numeric_feature_cols = [c for c in feature_cols if c not in numeric_cols_all]
    if non_numeric_feature_cols:
        st.warning(
            "Non-numeric selected feature(s) are excluded from statistical feature extraction: "
            + ", ".join(non_numeric_feature_cols)
        )
    numeric_processing_features = [c for c in feature_cols if c in numeric_cols_all]

    channels = st.multiselect(
        "Select Desired Channels/Signals for Processing",
        numeric_processing_features,
        default=numeric_processing_features,
    )
else:
    channels = []
    st.info("Upload a dataset to begin.")

st.markdown("---")

# =========================================================
# 3.1.7 DATA PREPROCESSING
# =========================================================
st.header("3.1.7 Data Preprocessing")

if df is not None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_signal_cols = numeric_feature_columns(df, st.session_state.label_col)

    # Duplicate rows can bias downstream summaries and PCA, so expose a simple
    # opt-in cleanup without changing the original uploaded file.
    st.subheader("Duplicate Row Check")
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count == 0:
        st.success("No duplicate rows detected.")
    else:
        st.warning(f"{duplicate_count:,} duplicate row(s) detected.")
        if st.button("Drop Duplicate Rows"):
            df = df.drop_duplicates().reset_index(drop=True)
            st.session_state.df = df
            st.success("Duplicate rows removed from the in-session dataset.")
            st.dataframe(df.head())

    st.subheader("Missing Value Check")
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if missing_summary.empty:
        st.success("No missing values detected.")
    else:
        st.warning("Missing values detected:")
        st.dataframe(missing_summary.rename("missing_count"))

        fill_strategy = st.selectbox(
            "Missing Value Handling Strategy",
            ["Do nothing", "Drop rows with missing values", "Fill with mean",
             "Fill with median", "Fill with mode", "Fill with zero", "Forward fill"]
        )
        if st.button("Apply Missing Value Handling"):
            if fill_strategy == "Drop rows with missing values":
                df = df.dropna()
            elif fill_strategy == "Fill with mean":
                if numeric_cols:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                else:
                    st.warning("No numeric columns are available for mean imputation.")
            elif fill_strategy == "Fill with median":
                if numeric_cols:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                else:
                    st.warning("No numeric columns are available for median imputation.")
            elif fill_strategy == "Fill with mode":
                df, skipped_mode_cols = fill_missing_with_mode(df)
                if skipped_mode_cols:
                    st.warning("Mode imputation skipped all-missing column(s): " + ", ".join(skipped_mode_cols))
            elif fill_strategy == "Fill with zero":
                df = df.fillna(0)
            elif fill_strategy == "Forward fill":
                df = df.ffill()
            elif fill_strategy == "Do nothing":
                st.info("No missing-value changes were applied.")
            st.session_state.df = df
            if fill_strategy != "Do nothing":
                st.success("Missing value handling applied.")
                st.dataframe(df.head())

    # Client requested removing user-controlled normalization/standardization from Tab 1.
    st.caption("Normalization and standardization controls were removed from preprocessing; data is not scaled in this section.")

st.markdown("---")

# =========================================================
# 3.1.2 / 3.1.3 FEATURE EXTRACTION & WINDOWING
# =========================================================
st.header("3.1.2 – 3.1.3 Feature Extraction (Statistical Features & Windowing)")

if df is not None and channels:
    window_size = st.number_input("Window Size", min_value=1, value=50)
    overlap = st.slider("Overlap Percentage", 0, 90, 50)

    if st.button("Extract Statistical Features"):
        if len(df) < window_size:
            st.warning("Window size is larger than the dataset, so no feature windows can be extracted.")
        else:
            step = max(1, int(window_size * (1 - overlap / 100)))
            features = []

            # Each window gets summary statistics for selected numeric signals;
            # labels are summarized separately so they do not become features.
            for start in range(0, len(df) - window_size + 1, step):
                window = df.iloc[start:start + window_size]
                stats = {"start_idx": start, "end_idx": start + window_size - 1}

                if st.session_state.label_col and st.session_state.label_col in window.columns:
                    label_window = window[st.session_state.label_col]
                    stats["label"] = safe_mode(label_window)

                for col in channels:
                    series = pd.to_numeric(window[col], errors="coerce")
                    stats[f"{col}_mean"] = series.mean()
                    stats[f"{col}_median"] = series.median()
                    stats[f"{col}_max"] = series.max()
                    stats[f"{col}_min"] = series.min()
                    stats[f"{col}_p2p"] = series.max() - series.min()
                    stats[f"{col}_var"] = series.var()
                    stats[f"{col}_std"] = series.std()
                    stats[f"{col}_rms"] = np.sqrt(np.mean(series.dropna() ** 2)) if series.notna().any() else np.nan
                    stats[f"{col}_skew"] = series.skew()
                    stats[f"{col}_kurt"] = series.kurt()

                features.append(stats)

            feat_df = pd.DataFrame(features)
            st.session_state.engineered_df = feat_df
            st.success(f"Feature Set Extracted - Shape: {feat_df.shape}")

if st.session_state.engineered_df is not None:
    st.dataframe(st.session_state.engineered_df.head())
elif df is not None and not channels:
    st.info("Select at least one channel/signal above to extract features.")

st.markdown("---")

# =========================================================
# 3.1.4 FEATURE SELECTION AND MANAGEMENT
# =========================================================
st.header("3.1.4 Feature Selection and Management")

if st.session_state.engineered_df is not None:
    feat_df = st.session_state.engineered_df
    non_feature_cols = [c for c in ["start_idx", "end_idx", "label"] if c in feat_df.columns]
    stat_cols = [c for c in feat_df.columns if c not in non_feature_cols]

    keep_cols = st.multiselect(
        "Add/Remove Features in Current Set", stat_cols, default=stat_cols
    )
    if st.button("Apply Feature Selection"):
        st.session_state.engineered_df = feat_df[non_feature_cols + keep_cols]
        st.success("Feature set updated.")
        st.dataframe(st.session_state.engineered_df.head())

    st.subheader("Save / Reload Feature Sets")
    save_col, load_col = st.columns(2)

    with save_col:
        set_name = st.text_input("Feature Set Name", value="feature_set_1")
        if st.button("Save Feature Set (this session)"):
            st.session_state.saved_feature_sets[set_name] = st.session_state.engineered_df.copy()
            st.success(f"Saved '{set_name}' for this session.")

        st.download_button(
            "Download Feature Set (.pkl) for Later Reload",
            to_pickle_bytes(st.session_state.engineered_df),
            file_name=f"{set_name}.pkl",
        )

    with load_col:
        if st.session_state.saved_feature_sets:
            reload_choice = st.selectbox(
                "Reload Session-Saved Set", list(st.session_state.saved_feature_sets.keys())
            )
            if st.button("Reload Selected Set"):
                st.session_state.engineered_df = st.session_state.saved_feature_sets[reload_choice].copy()
                st.success(f"Reloaded '{reload_choice}'.")

        reload_file = st.file_uploader("Or Reload Feature Set from .pkl File", type=["pkl"], key="reload_pkl")
        if reload_file is not None and st.button("Load Uploaded Feature Set"):
            st.session_state.engineered_df = pickle.load(reload_file)
            st.success("Feature set loaded from file.")
else:
    st.info("Extract a feature set above to manage/save/reload features.")

st.markdown("---")

# =========================================================
# 3.1.5 VISUALIZATION
# =========================================================
st.header("3.1.5 Visualization")

if df is not None:
    data_source = st.radio(
        "Data Source for Visualization",
        ["Raw Data", "Engineered Features"] if st.session_state.engineered_df is not None else ["Raw Data"],
        horizontal=True,
    )
    viz_df = df if data_source == "Raw Data" else st.session_state.engineered_df
    viz_numeric = viz_df.select_dtypes(include=[np.number]).columns.tolist()

    color_col = None
    if data_source == "Raw Data" and st.session_state.label_col:
        color_col = st.session_state.label_col
    elif data_source == "Engineered Features" and "label" in viz_df.columns:
        color_col = "label"
    can_color_by_label = (
        color_col is not None
        and color_col in viz_df.columns
        and is_label_like(viz_df[color_col])
    )

    viz_signal_numeric = [
        c for c in viz_numeric
        if c != color_col and not is_metadata_column(c)
    ]
    viz_plot_defaults = viz_signal_numeric if viz_signal_numeric else viz_numeric

    # One shared config keeps all visualization controls synchronized across plots.
    with st.expander("Visualization controls", expanded=True):
        ctrl_cols = st.columns(4, vertical_alignment="bottom")
        with ctrl_cols[0]:
            plot_width = st.slider("Plot width", 5, 12, 7, key="viz_plot_width")
        with ctrl_cols[1]:
            plot_height = st.slider("Plot height", 3, 8, 4, key="viz_plot_height")
        with ctrl_cols[2]:
            zoom_pct = st.slider("Zoom level (%)", 50, 200, 100, 10, key="viz_zoom_pct")
        with ctrl_cols[3]:
            axis_scale = st.selectbox("Axis scale", ["Linear", "Log", "Symmetric log"], key="viz_axis_scale")
        hist_bins = st.slider("Histogram bins", 5, 200, 30, key="viz_hist_bins")
        st.caption("Below 100% zooms in; above 100% zooms out. Log scale is applied only when the selected axis data is strictly positive.")
    viz_config = build_plot_config(plot_width, plot_height, zoom_pct, axis_scale, hist_bins)

    if not viz_numeric:
        st.warning("No numeric columns are available for visualization.")
    else:
        # Data-quality visuals appear before PCA so missingness, label balance,
        # feature spread, and outliers are visible before dimensionality reduction.
        st.subheader("Missing-Values Overview")
        viz_missing = missing_value_counts(viz_df)
        viz_missing = viz_missing[viz_missing > 0]
        if viz_missing.empty:
            st.success("No missing values detected in the selected visualization data.")
        else:
            fig_missing, ax_missing = plt.subplots(figsize=viz_config["figsize"])
            viz_missing.sort_values().plot(kind="barh", ax=ax_missing)
            ax_missing.set_xlabel("Missing Values")
            ax_missing.set_ylabel("Column")
            ax_missing.set_title("Missing Values by Column")
            apply_plot_controls(ax_missing, viz_config, x_values=viz_missing.values, apply_x=True, apply_y=False)
            st.pyplot(fig_missing)
            st.caption("Columns with more missing values may need imputation or exclusion before PCA.")
            download_fig_button(fig_missing, "dl_missing_values", "missing_values.png")

        if can_color_by_label:
            st.subheader("Target / Class Distribution")
            target_counts = viz_df[color_col].value_counts(dropna=False).sort_index()
            fig_target, ax_target = plt.subplots(figsize=viz_config["figsize"])
            target_counts.plot(kind="bar", ax=ax_target)
            ax_target.set_xlabel(color_col)
            ax_target.set_ylabel("Count")
            ax_target.set_title(f"Distribution of {color_col}")
            ax_target.tick_params(axis="x", rotation=0)
            apply_plot_controls(ax_target, viz_config, y_values=target_counts.values, apply_x=False, apply_y=True)
            st.pyplot(fig_target)
            st.caption("Class balance affects interpretation of PCA coloring and later supervised learning.")
            download_fig_button(fig_target, "dl_target_dist", "target_distribution.png")
        elif color_col and color_col in viz_df.columns:
            st.info("Selected label has many unique numeric values, so it is not plotted as a class distribution.")

        if viz_signal_numeric:
            st.subheader("Selected Feature Distributions")
            dist_cols = st.multiselect(
                "Select Numeric Features for Distribution Plots",
                viz_signal_numeric,
                default=viz_signal_numeric[:min(4, len(viz_signal_numeric))],
                key="dist_cols",
            )
            if dist_cols:
                n_cols = min(2, len(dist_cols))
                n_rows = int(np.ceil(len(dist_cols) / n_cols))
                fig_dist, axes = plt.subplots(n_rows, n_cols, figsize=(viz_config["width"] * n_cols, viz_config["height"] * n_rows))
                axes = np.atleast_1d(axes).ravel()
                for ax, col in zip(axes, dist_cols):
                    series = pd.to_numeric(viz_df[col], errors="coerce").dropna()
                    ax.hist(series, bins=viz_config["hist_bins"], alpha=0.8)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {col}")
                    apply_plot_controls(ax, viz_config, x_values=series, apply_x=True, apply_y=False)
                for ax in axes[len(dist_cols):]:
                    ax.axis("off")
                fig_dist.tight_layout()
                st.pyplot(fig_dist)
                st.caption("Distributions reveal skew, concentration, and unusual value ranges before PCA.")
                download_fig_button(fig_dist, "dl_dist", "selected_feature_distributions.png")

            st.subheader("Correlation Heatmap")
            corr_candidates = [
                c for c in viz_signal_numeric
                if pd.to_numeric(viz_df[c], errors="coerce").nunique(dropna=True) > 1
            ]
            corr_cols = st.multiselect(
                "Select Numeric Features for Correlation",
                corr_candidates,
                default=corr_candidates[:min(10, len(corr_candidates))],
                key="corr_cols",
            )
            if len(corr_cols) >= 2:
                corr = viz_df[corr_cols].corr()
                fig_corr, ax_corr = plt.subplots(figsize=viz_config["figsize"])
                im = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
                ax_corr.set_xticks(range(len(corr_cols)))
                ax_corr.set_yticks(range(len(corr_cols)))
                ax_corr.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
                ax_corr.set_yticklabels(corr_cols, fontsize=8)
                ax_corr.set_title("Correlation Heatmap")
                fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
                fig_corr.tight_layout()
                st.pyplot(fig_corr)
                st.caption("Highly correlated features may carry overlapping information before PCA.")
                download_fig_button(fig_corr, "dl_corr", "correlation_heatmap.png")
            else:
                st.info("Select at least two non-constant numeric features to show a correlation heatmap.")

            st.subheader("Feature Variability")
            variability = viz_df[viz_signal_numeric].replace([np.inf, -np.inf], np.nan).std(numeric_only=True).dropna()
            if variability.empty:
                st.info("No valid numeric variability values are available.")
            else:
                top_variability = variability.sort_values(ascending=False).head(20)
                fig_var_feat, ax_var_feat = plt.subplots(figsize=viz_config["figsize"])
                top_variability.sort_values().plot(kind="barh", ax=ax_var_feat)
                ax_var_feat.set_xlabel("Standard Deviation")
                ax_var_feat.set_ylabel("Feature")
                ax_var_feat.set_title("Feature Standard Deviation")
                apply_plot_controls(ax_var_feat, viz_config, x_values=top_variability.values, apply_x=True, apply_y=False)
                st.pyplot(fig_var_feat)
                st.caption("Low-variability features may contribute little to PCA or later modeling.")
                download_fig_button(fig_var_feat, "dl_feature_std", "feature_standard_deviation.png")

            st.subheader("Outlier Overview")
            box_cols = st.multiselect(
                "Select Numeric Features for Box Plots",
                viz_signal_numeric,
                default=viz_signal_numeric[:min(6, len(viz_signal_numeric))],
                key="box_cols",
            )
            if box_cols:
                box_data = [
                    pd.to_numeric(viz_df[c], errors="coerce").dropna()
                    for c in box_cols
                ]
                valid_box = [(c, s) for c, s in zip(box_cols, box_data) if not s.empty]
                if valid_box:
                    fig_box, ax_box = plt.subplots(figsize=(max(viz_config["width"], 1.2 * len(valid_box)), viz_config["height"]))
                    ax_box.boxplot([s for _, s in valid_box], tick_labels=[c for c, _ in valid_box], showfliers=True)
                    ax_box.set_ylabel("Value")
                    ax_box.set_title("Outlier Overview by Feature")
                    apply_plot_controls(ax_box, viz_config, y_values=np.concatenate([s.to_numpy() for _, s in valid_box]), apply_x=False, apply_y=True)
                    ax_box.tick_params(axis="x", rotation=45)
                    fig_box.tight_layout()
                    st.pyplot(fig_box)
                    st.caption("Box plots highlight spread and potential outliers without assigning causes.")
                    download_fig_button(fig_box, "dl_box", "outlier_overview.png")
                else:
                    st.info("Selected features do not contain valid numeric values for box plots.")
        else:
            st.info("No numeric signal features remain after excluding labels and metadata.")

    # ---- Time-series plot ----
    st.subheader("Time-Series Plot")
    ts_cols = st.multiselect("Select Signal(s) to Plot", viz_numeric, default=viz_plot_defaults[:1])
    if ts_cols:
        fig_ts, ax_ts = plt.subplots(figsize=viz_config["figsize"])
        for c in ts_cols:
            ax_ts.plot(viz_df.index, viz_df[c], label=c, linewidth=1)
        ax_ts.set_xlabel("Index")
        ax_ts.set_ylabel("Value")
        ax_ts.set_title("Time-Series Plot")
        apply_plot_controls(ax_ts, viz_config, x_values=viz_df.index, y_values=viz_df[ts_cols].to_numpy(), apply_x=True, apply_y=True)
        ax_ts.legend()
        st.pyplot(fig_ts)
        download_fig_button(fig_ts, "dl_ts", "time_series_plot.png")

    # ---- Histogram ----
    st.subheader("Histogram")
    if viz_numeric:
        hist_col = st.selectbox("Select Column for Histogram", viz_numeric, key="hist_col")
        if hist_col:
            fig_hist, ax_hist = plt.subplots(figsize=viz_config["figsize"])
            if can_color_by_label:
                for label_val, group in viz_df.groupby(color_col):
                    ax_hist.hist(group[hist_col].dropna(), bins=viz_config["hist_bins"], alpha=0.6, label=str(label_val))
                ax_hist.legend(title=color_col)
            else:
                ax_hist.hist(viz_df[hist_col].dropna(), bins=viz_config["hist_bins"], alpha=0.8)
            ax_hist.set_xlabel(hist_col)
            ax_hist.set_ylabel("Frequency")
            ax_hist.set_title(f"Histogram of {hist_col}")
            apply_plot_controls(ax_hist, viz_config, x_values=viz_df[hist_col], apply_x=True, apply_y=False)
            st.pyplot(fig_hist)
            download_fig_button(fig_hist, "dl_hist", "histogram.png")

    # ---- 2D / 3D scatter ----
    st.subheader("2D and 3D Scatter Plots")
    scatter_dim = st.radio("Scatter Type", ["2D", "3D"], horizontal=True)

    if scatter_dim == "2D" and len(viz_numeric) >= 2:
        sx = st.selectbox("X-axis", viz_numeric, index=0, key="sx2d")
        sy = st.selectbox("Y-axis", viz_numeric, index=1, key="sy2d")
        fig_sc, ax_sc = plt.subplots(figsize=viz_config["figsize"])
        if can_color_by_label:
            for label_val, group in viz_df.groupby(color_col):
                ax_sc.scatter(group[sx], group[sy], label=str(label_val), alpha=0.7)
            ax_sc.legend(title=color_col)
        else:
            ax_sc.scatter(viz_df[sx], viz_df[sy], alpha=0.7)
        ax_sc.set_xlabel(sx)
        ax_sc.set_ylabel(sy)
        ax_sc.set_title("2D Scatter Plot")
        apply_plot_controls(ax_sc, viz_config, x_values=viz_df[sx], y_values=viz_df[sy], apply_x=True, apply_y=True)
        st.pyplot(fig_sc)
        download_fig_button(fig_sc, "dl_sc2d", "scatter_2d.png")

    elif scatter_dim == "3D" and len(viz_numeric) >= 3:
        sx3 = st.selectbox("X-axis", viz_numeric, index=0, key="sx3d")
        sy3 = st.selectbox("Y-axis", viz_numeric, index=1, key="sy3d")
        sz3 = st.selectbox("Z-axis", viz_numeric, index=2, key="sz3d")
        fig3d = plt.figure(figsize=(viz_config["width"], viz_config["height"] + 1))
        ax3d = fig3d.add_subplot(111, projection="3d")
        if can_color_by_label:
            for label_val, group in viz_df.groupby(color_col):
                ax3d.scatter(group[sx3], group[sy3], group[sz3], label=str(label_val), alpha=0.7)
            ax3d.legend(title=color_col)
        else:
            ax3d.scatter(viz_df[sx3], viz_df[sy3], viz_df[sz3], alpha=0.7)
        ax3d.set_xlabel(sx3)
        ax3d.set_ylabel(sy3)
        ax3d.set_zlabel(sz3)
        ax3d.set_title("3D Scatter Plot")
        apply_3d_zoom_controls(ax3d, viz_config, viz_df[sx3], viz_df[sy3], viz_df[sz3])
        st.pyplot(fig3d)
        download_fig_button(fig3d, "dl_sc3d", "scatter_3d.png")
    else:
        st.info("Need at least 2 (2D) or 3 (3D) numeric columns for scatter plots.")

    # ---- Multi-feature comparison (scatter matrix) ----
    st.subheader("Multi-Feature Comparison Plot")
    multi_cols = st.multiselect(
        "Select Features to Compare", viz_numeric, default=viz_plot_defaults[:min(4, len(viz_plot_defaults))]
    )
    if len(multi_cols) >= 2:
        fig_mat, axes = plt.subplots(len(multi_cols), len(multi_cols), figsize=(max(viz_config["width"], 2.6 * len(multi_cols)), max(viz_config["height"], 2.6 * len(multi_cols))))
        groups = list(viz_df.groupby(color_col)) if can_color_by_label else [(None, viz_df)]
        for i, ci in enumerate(multi_cols):
            for j, cj in enumerate(multi_cols):
                ax = axes[i][j] if len(multi_cols) > 1 else axes
                if i == j:
                    for label_val, group in groups:
                        ax.hist(group[ci].dropna(), bins=viz_config["hist_bins"], alpha=0.6)
                else:
                    for label_val, group in groups:
                        ax.scatter(group[cj], group[ci], s=8, alpha=0.6)
                if i == j:
                    apply_plot_controls(ax, viz_config, x_values=viz_df[ci], apply_x=True, apply_y=False)
                else:
                    apply_plot_controls(ax, viz_config, x_values=viz_df[cj], y_values=viz_df[ci], apply_x=True, apply_y=True)
                if j == 0:
                    ax.set_ylabel(ci, fontsize=8)
                if i == len(multi_cols) - 1:
                    ax.set_xlabel(cj, fontsize=8)
                ax.tick_params(labelsize=6)
        fig_mat.suptitle("Multi-Feature Comparison (Scatter Matrix)")
        fig_mat.tight_layout()
        st.pyplot(fig_mat)
        download_fig_button(fig_mat, "dl_multi", "multi_feature_comparison.png")

    # After visualization, users can retain only the columns they judge important
    # and download that smaller feature set without changing the active dataset.
    st.subheader("Retain Important Features After Visualization")
    if viz_signal_numeric:
        feature_scores = viz_df[viz_signal_numeric].replace([np.inf, -np.inf], np.nan).std(numeric_only=True).dropna()
        default_important = feature_scores.sort_values(ascending=False).head(min(8, len(feature_scores))).index.tolist()
        if not default_important:
            default_important = viz_signal_numeric[:min(8, len(viz_signal_numeric))]
        important_cols = st.multiselect(
            "Select important features to retain",
            viz_signal_numeric,
            default=default_important,
            key="important_feature_cols",
            help="Use the visualizations above to keep only the features that appear useful for download."
        )
        include_label_export = st.checkbox(
            "Include selected label/class column in retained download",
            value=bool(can_color_by_label),
            key="include_label_in_important_export",
        )
        if st.button("Prepare Retained Feature Set"):
            retained_df = build_important_feature_subset(viz_df, important_cols, color_col, include_label_export)
            st.session_state.selected_important_df = retained_df
            st.success(f"Retained feature set prepared - Shape: {retained_df.shape}")
        if st.session_state.selected_important_df is not None:
            retained_df = st.session_state.selected_important_df
            st.dataframe(retained_df.head())
            r1, r2, r3 = st.columns(3)
            with r1:
                st.download_button(
                    "Download Retained CSV",
                    to_csv_bytes(retained_df),
                    file_name="retained_important_features.csv",
                    mime="text/csv",
                )
            with r2:
                st.download_button(
                    "Download Retained NumPy",
                    to_npy_bytes(retained_df),
                    file_name="retained_important_features.npy",
                )
            with r3:
                st.download_button(
                    "Download Retained Pickle",
                    to_pickle_bytes(retained_df),
                    file_name="retained_important_features.pkl",
                )
    else:
        st.info("No numeric signal features are available to retain from this visualization source.")
else:
    st.info("Upload a dataset to enable visualizations.")

st.markdown("---")

# =========================================================
# 3.1.6 PCA AND STATISTICAL ANALYSIS
# =========================================================
st.header("3.1.6 PCA and Statistical Analysis")

if df is not None:
    pca_source = st.radio(
        "Data Source for PCA",
        ["Raw Data", "Engineered Features"] if st.session_state.engineered_df is not None else ["Raw Data"],
        horizontal=True, key="pca_source",
    )
    pca_df_full = df if pca_source == "Raw Data" else st.session_state.engineered_df
    pca_label = st.session_state.label_col if pca_source == "Raw Data" else (
        "label" if "label" in pca_df_full.columns else None
    )

    pca_ready_df, pca_report = prepare_pca_feature_matrix(pca_df_full, pca_label)
    raw_numeric_cols = pca_df_full.select_dtypes(include=[np.number]).columns.tolist()
    excluded_label_or_metadata = [
        c for c in raw_numeric_cols
        if c not in pca_report["candidate_cols"] and (c == pca_label or is_metadata_column(c))
    ]

    # The PCA audit makes preprocessing explicit instead of silently filling or
    # fitting on unsuitable fields such as labels, IDs, timestamps, or constants.
    st.subheader("PCA Preparation Summary")
    prep_cols = st.columns(4)
    prep_cols[0].metric("Candidate Features", f"{len(pca_report['candidate_cols']):,}")
    prep_cols[1].metric("PCA Features Used", f"{pca_ready_df.shape[1]:,}")
    prep_cols[2].metric("Rows Used", f"{pca_ready_df.shape[0]:,}")
    prep_cols[3].metric("Missing Values Imputed", f"{pca_report['missing_values_before_imputation']:,}")

    if excluded_label_or_metadata:
        st.caption(
            "Excluded label/metadata column(s) from PCA features: "
            + ", ".join(excluded_label_or_metadata)
        )
    if pca_report["constant_cols"]:
        st.warning("Constant feature(s) excluded from PCA: " + ", ".join(pca_report["constant_cols"]))
    if pca_report["near_constant_cols"]:
        st.warning("Near-constant feature(s) excluded from PCA: " + ", ".join(pca_report["near_constant_cols"]))
    if pca_report["all_missing_cols"]:
        st.warning("All-missing feature(s) excluded from PCA: " + ", ".join(pca_report["all_missing_cols"]))
    if pca_report["missing_values_before_imputation"] > 0:
        st.info("Missing PCA feature values were imputed with each feature's median.")
    if pca_report["rows_dropped_after_imputation"] > 0:
        st.warning(f"{pca_report['rows_dropped_after_imputation']:,} row(s) were dropped because missing values remained after imputation.")

    max_components = min(10, pca_ready_df.shape[0], pca_ready_df.shape[1])
    if pca_ready_df.shape[1] < 2:
        st.warning("PCA needs at least two valid numeric feature columns after exclusions.")
    elif pca_ready_df.shape[0] < 2:
        st.warning("PCA needs at least two valid rows after preprocessing.")
    else:
        n_components = st.slider("Number of Principal Components", 2, max_components, 2)

        if st.button("Run PCA"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(pca_ready_df)
            pca = PCA(n_components=max_components)
            X_pca = pca.fit_transform(X_scaled)

            fig_pca, ax_pca = plt.subplots(figsize=(7, 5))
            if pca_label and pca_label in pca_df_full.columns and is_label_like(pca_df_full[pca_label]):
                plot_df = pd.DataFrame({
                    "PC1": X_pca[:, 0],
                    "PC2": X_pca[:, 1],
                    pca_label: pca_df_full.loc[pca_ready_df.index, pca_label].values,
                })
                for lv, group in plot_df.groupby(pca_label, dropna=False):
                    ax_pca.scatter(group["PC1"], group["PC2"], alpha=0.7, label=str(lv))
                ax_pca.legend(title=pca_label)
            else:
                ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
            ax_pca.set_xlabel("PC1")
            ax_pca.set_ylabel("PC2")
            ax_pca.set_title("PCA Scatter Plot")
            st.pyplot(fig_pca)
            st.caption(
                "PC1 is the direction of greatest variance in the PCA-ready numeric features; "
                "PC2 is the next independent direction of variance."
            )
            download_fig_button(fig_pca, "dl_pca", "pca_scatter.png")

            explained = pca.explained_variance_ratio_
            cumulative = np.cumsum(explained)
            component_numbers = np.arange(1, max_components + 1)

            fig_var, ax_var = plt.subplots(figsize=(7, 4))
            ax_var.bar(component_numbers, explained, alpha=0.75, label="Explained variance")
            ax_var.plot(component_numbers, cumulative, marker="o", color="black", label="Cumulative variance")
            ax_var.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% threshold")
            ax_var.set_xlabel("Principal Component")
            ax_var.set_ylabel("Variance Ratio")
            ax_var.set_title("Explained and Cumulative Variance")
            ax_var.set_xticks(component_numbers)
            ax_var.set_ylim(0, min(1.05, max(1.0, cumulative.max() + 0.05)))
            ax_var.legend()
            st.pyplot(fig_var)
            download_fig_button(fig_var, "dl_pca_var", "pca_explained_variance.png")

            variance_table = pd.DataFrame({
                "Principal Component": [f"PC{i}" for i in component_numbers],
                "Explained Variance Ratio": explained,
                "Cumulative Explained Variance": cumulative,
            })
            st.dataframe(variance_table)

            reached_threshold = np.where(cumulative >= 0.90)[0]
            if len(reached_threshold) > 0:
                recommended_components = int(reached_threshold[0] + 1)
                st.success(
                    f"Recommendation: use at least {recommended_components} component(s) "
                    "to capture about 90% of the feature variance."
                )
            else:
                st.info(
                    f"The first {max_components} component(s) do not reach 90% cumulative explained variance; "
                    "review more components or keep the PCA output exploratory."
                )

            top_component_count = min(n_components, 2)
            loading_cols = [f"PC{i}" for i in range(1, top_component_count + 1)]
            loadings = pd.DataFrame(
                pca.components_[:top_component_count].T,
                index=pca_ready_df.columns,
                columns=loading_cols,
            )
            top_features = []
            for pc in loading_cols:
                top_features.extend(loadings[pc].abs().sort_values(ascending=False).head(10).index.tolist())
            top_features = list(dict.fromkeys(top_features))
            if top_features:
                st.subheader("Top PCA Feature Contributions")
                loading_view = loadings.loc[top_features].copy()
                for pc in loading_cols:
                    loading_view[f"{pc} Absolute Contribution"] = loading_view[pc].abs()
                st.dataframe(loading_view)
                st.caption("Larger absolute loadings contribute more strongly to the displayed principal component.")

    st.subheader("Feature Importance Analysis")
    fi_label = pca_label
    if fi_label and fi_label in pca_df_full.columns:
        if pca_ready_df.empty:
            st.info("No valid numeric features are available for feature importance analysis.")
        else:
            fi_method = st.selectbox("Method", ["Correlation Analysis", "Mutual Information"])
            if st.button("Compute Feature Importance"):
                y_raw = pca_df_full.loc[pca_ready_df.index, fi_label]
                valid_target = y_raw.notna()
                X_fi = pca_ready_df.loc[valid_target]
                y_raw = y_raw.loc[valid_target]

                if X_fi.shape[0] < 2 or y_raw.nunique(dropna=True) < 2:
                    st.warning("Feature importance needs at least two rows and two target values.")
                else:
                    is_categorical = is_label_like(y_raw)

                    if fi_method == "Correlation Analysis":
                        if is_categorical:
                            st.warning("Label appears categorical; using label-encoded values for correlation.")
                            y = pd.factorize(y_raw)[0]
                        else:
                            y = pd.to_numeric(y_raw, errors="coerce").values
                        corr_vals = X_fi.apply(
                            lambda col: np.corrcoef(col, y)[0, 1]
                            if col.std() > 0 and np.nanstd(y) > 0 else np.nan
                        ).dropna()
                        if corr_vals.empty:
                            st.warning("No valid correlations could be computed.")
                        else:
                            fig_fi, ax_fi = plt.subplots(figsize=(6, max(3, 0.3 * len(corr_vals))))
                            corr_vals.sort_values().plot(kind="barh", ax=ax_fi)
                            ax_fi.set_title(f"Correlation with '{fi_label}'")
                            st.pyplot(fig_fi)
                            download_fig_button(fig_fi, "dl_fi_corr", "feature_importance_correlation.png")

                    else:
                        y = pd.factorize(y_raw)[0] if is_categorical else pd.to_numeric(y_raw, errors="coerce").values
                        mi_func = mutual_info_classif if is_categorical else mutual_info_regression
                        mi_vals = mi_func(X_fi, y, random_state=0)
                        mi_series = pd.Series(mi_vals, index=X_fi.columns).sort_values()
                        fig_fi, ax_fi = plt.subplots(figsize=(6, max(3, 0.3 * len(mi_series))))
                        mi_series.plot(kind="barh", ax=ax_fi)
                        ax_fi.set_title(f"Mutual Information with '{fi_label}'")
                        st.pyplot(fig_fi)
                        download_fig_button(fig_fi, "dl_fi_mi", "feature_importance_mutual_info.png")
    else:
        st.info("Select a label/class column in Section 3.1.1 to enable feature importance analysis.")
else:
    st.info("Upload a dataset to enable PCA and statistical analysis.")

st.markdown("---")

# =========================================================
# 3.1.8 EXPORT CAPABILITY
# =========================================================
st.header("3.1.8 Export Capability")

export_options = ["Raw / Preprocessed Data", "Engineered Feature Set"]
if st.session_state.selected_important_df is not None:
    export_options.append("Retained Important Features")

export_target = st.radio(
    "Dataset to Export",
    export_options,
    horizontal=True,
)
if export_target == "Raw / Preprocessed Data":
    export_df = df
elif export_target == "Engineered Feature Set":
    export_df = st.session_state.engineered_df
else:
    export_df = st.session_state.selected_important_df

if export_df is not None:
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "Download as CSV",
            to_csv_bytes(export_df),
            file_name="processed_data.csv",
            mime="text/csv",
        )
    with e2:
        st.download_button(
            "Download as NumPy (.npy)",
            to_npy_bytes(export_df),
            file_name="processed_data.npy",
        )
    with e3:
        st.download_button(
            "Download as Pickle (.pkl)",
            to_pickle_bytes(export_df),
            file_name="processed_data.pkl",
        )
else:
    st.info("Nothing available to export yet for this selection.")

st.markdown("---")
# Logos were moved to the bottom per the client comment, keeping the top area
# compact while still preserving project branding.
footer_left, footer_mid, footer_right = st.columns([1, 2, 1], vertical_alignment="center")
with footer_left:
    st.image("assets/icaav_logo.png", width=110)
with footer_mid:
    st.caption("iCAAV Core - Advanced Biomechatronics and Locomotion Laboratory - Carleton University")
with footer_right:
    st.image("assets/carleton_logo.png", width=110)


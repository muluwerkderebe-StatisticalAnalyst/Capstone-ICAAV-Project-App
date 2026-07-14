import streamlit as st
from theme import apply_theme

apply_theme()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from io import BytesIO
 
from data_utils import build_numeric_view
from udp_receiver import UDPReceiver

 
# -----------------------------
# BRANDING HEADER
# -----------------------------
col1, col2, col3 = st.columns([1, 3, 1])
 
with col1:
    try:
        st.image("assets/icaav_logo.png", width=100)
    except Exception:
        st.markdown("**iCAAV**")
 
with col2:
    st.markdown(
        """
        <div class="icaav-page-title">
            Tab 4 — Real-Time Testing, Visualization, and Annotation
        </div>

        <div class="icaav-page-subtitle">
            Intelligent Connected Assistive & Autonomous Vehicles (iCAAV) Core
            <br>Advanced Biomechatronics and Locomotion Laboratory
            <br>Carleton University
        </div>
        """,
        unsafe_allow_html=True,
    )
 
with col3:
    try:
        st.image("assets/carleton_logo.png", width=100)
    except Exception:
        st.markdown("**Carleton**")
 
st.markdown("---")
 
# -----------------------------
# SESSION STATE INIT
# -----------------------------
for key, default in {
    "running": False,
    "paused": False,
    "playback_mode": "stopped",
    "current_index": 0,
    "annotation_data": [],
    "buffer": {},
    "prev_jump": 0,
    "jump_target": 0,
    "uploaded_file_name": None,
    "df": None,
    "model": None,
    "model_name": None,
    "predicted_label": None,
    "predicted_labels_history": [],
    "active_value_cols": [],
    "active_speed": 5,
    "active_window_size": 100,
    # --- UDP additions ---
    "data_source": "Recorded Playback",
    "live_last_row": {},
    "udp_index": 0,
    "pred_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
 
 
# -----------------------------
# UDP RECEIVER (single shared listener across reruns)
# -----------------------------
@st.cache_resource
def get_udp_receiver(port: int):
    r = UDPReceiver(port=port)
    r.start()
    return r
 
 
# -----------------------------
# SECTION 1 — LOAD DATASET
# -----------------------------
st.header("1. Load Dataset")
 
uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
 
if uploaded is not None:
    if uploaded.name != st.session_state.get("uploaded_file_name"):
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.playback_mode = "stopped"
        st.session_state.current_index = 0
        st.session_state.prev_jump = 0
        st.session_state.jump_target = 0
        st.session_state.buffer = {}
        st.session_state.annotation_data = []
        st.session_state.predicted_labels_history = []
        st.session_state.active_value_cols = []
        st.session_state.uploaded_file_name = uploaded.name
        st.session_state.live_last_row = {}
        st.session_state.udp_index = 0
 
if st.session_state.df is None:
    st.info("Upload a dataset (CSV) to begin. In Live UDP mode the CSV defines the "
            "column layout, model features, and chart settings; live values then arrive over UDP.")
    st.stop()
 
df = st.session_state.df
if df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()
 
numeric_df = build_numeric_view(df)
numeric_cols = [c for c in numeric_df.columns if numeric_df[c].notna().any()]
 
st.success(f"Dataset loaded — {df.shape[0]} rows x {df.shape[1]} columns")
st.caption("Non-numeric values were coerced to numeric form so this page can work with mixed-type CSVs.")
st.dataframe(df.head(3), use_container_width=True)
 
time_col = st.selectbox("Select X-Axis", ["Row Index"] + df.columns.tolist())
 
preferred = ["V", "str", "ax", "ay", "HeartRate", "bra", "LO", "HA"]
smart_default = [c for c in preferred if c in numeric_cols]
if len(smart_default) < 6:
    extras = [c for c in numeric_cols if c not in smart_default]
    smart_default += extras[: max(0, 6 - len(smart_default))]
if not smart_default:
    smart_default = numeric_cols[:6]
 
value_cols = st.multiselect(
    "Select Columns to Plot",
    numeric_cols,
    default=smart_default[:6]
)
 
if len(value_cols) == 0:
    st.warning("Select at least one column to continue.")
    st.stop()
 
if len(value_cols) > 6:
    st.warning("More than 6 selected — only the first 6 will be plotted.")
    value_cols = value_cols[:6]
 
CHART_TYPES = ["Line", "Scatter", "Histogram", "Area", "Bar", "Box"]
 
graph_type_global = st.radio(
    "Default Chart Type (applies to all)",
    CHART_TYPES,
    horizontal=True,
    key="global_chart_type"
)
 
st.markdown("**Chart Type per Column:**")
per_col_type = {}
type_cols = st.columns(len(value_cols))
for j, col_name in enumerate(value_cols):
    override_key = f"chart_type_{col_name}"
    global_sync_key = f"last_global_{col_name}"
    if st.session_state.get(global_sync_key) != graph_type_global:
        st.session_state[override_key] = graph_type_global
        st.session_state[global_sync_key] = graph_type_global
    per_col_type[col_name] = type_cols[j].selectbox(col_name, CHART_TYPES, key=override_key)
 
label_named_cols = [
    c for c in df.columns
    if c.lower() in ["label", "class", "labels", "classes", "target"]
    and c not in value_cols
]
if label_named_cols:
    class_col_options = ["None"] + label_named_cols
else:
    class_col_options = ["None"] + [
        c for c in df.columns
        if df[c].nunique(dropna=True) <= 10 and c not in value_cols
    ]
 
color_by = st.selectbox("Color Plots by Class", class_col_options)
 
st.markdown("---")
 
# -----------------------------
# SECTION 2 — LOAD TRAINED MODEL
# -----------------------------
st.header("2. Load Trained Model")
 
model_file = st.file_uploader("Upload trained model (.pkl)", type=["pkl"])
 
if model_file is not None:
    try:
        st.session_state.model = joblib.load(BytesIO(model_file.read()))
        st.session_state.model_name = model_file.name
        st.success(f"Model loaded: **{model_file.name}**")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
 
if st.session_state.model is not None:
    st.info(f"Active model: **{st.session_state.model_name}**")
 
    # Prefer the feature names the model was trained on. This guarantees the
    # live/playback input matches what the model expects (name, count, order),
    # removing the risk of the user picking mismatched columns.
    model_expected = None
    if hasattr(st.session_state.model, "feature_names_in_"):
        model_expected = [str(c) for c in st.session_state.model.feature_names_in_]
 
    if model_expected:
        model_feature_cols = model_expected
        st.success(
            f"Model expects {len(model_feature_cols)} feature(s), taken from the trained model: "
            + ", ".join(model_feature_cols)
        )
        missing = [c for c in model_feature_cols if c not in numeric_cols]
        if missing:
            st.warning(
                "The loaded dataset/stream is missing feature(s) the model needs: "
                + ", ".join(missing)
                + ". Predictions may be invalid until these columns are present."
            )
    else:
        st.caption(
            "This model did not record its training feature names. Select the exact "
            "columns it was trained on, in the same order."
        )
        model_feature_cols = st.multiselect(
            "Select Model Feature Columns",
            numeric_cols,
            default=value_cols
        )
 
    class_map = st.text_input(
        "Class Label Map",
        value="0=Safe, 1=Aggressive, 2=Distracted"
    )
    label_map = {}
    try:
        for item in class_map.split(","):
            k, v = item.strip().split("=")
            label_map[int(k.strip())] = v.strip()
    except Exception:
        label_map = {}
else:
    st.info("No model loaded — playback will run without classification.")
    model_feature_cols = value_cols
    label_map = {}
 
st.markdown("---")
# -----------------------------
# SECTION 3 — DATA SOURCE & PLAYBACK CONTROLS
# -----------------------------
st.header("3. Data Source & Playback Controls")
 
data_source = st.radio(
    "Data Source",
    ["Recorded Playback", "Live UDP Stream"],
    horizontal=True,
    key="data_source"
)
 
udp_port = 5005
 
if data_source == "Live UDP Stream":
    receiver = get_udp_receiver(udp_port)
    if receiver.bind_error:
        st.error(f"Could not bind UDP port {udp_port}: {receiver.bind_error}")
    else:
        st.success(f"Listening for UDP packets on port {udp_port} — "
                   f"packets received so far: {receiver.count()}")
    st.caption("Run the sender in a terminal, e.g.:  "
               "python udp_sender.py --csv RawDataPoints.csv --cols "
               + ",".join(value_cols) + " --port 5005 --rate 10")
 
speed = st.slider("Playback Speed (rows per second)", 1, 20, 5, key="playback_speed")
window_size = st.slider("Chart Window (last N rows to display)", 50, 500, 100, key="playback_window")
 
# Jump-to-row only applies to recorded playback
if data_source == "Recorded Playback":
    jump_target = st.number_input(
        "Jump to Row",
        min_value=0,
        max_value=max(len(df) - 1, 0),
        value=int(st.session_state.jump_target),
        step=1,
        key="jump_row_input"
    )
    jump_btn = st.button("Jump to Selected Row", use_container_width=True)
 
    if jump_btn:
        st.session_state.jump_target = int(jump_target)
        st.session_state.prev_jump = int(jump_target)
        st.session_state.current_index = int(jump_target)
        st.session_state.paused = True
        st.session_state.running = False
        st.session_state.playback_mode = "paused"
        st.session_state.buffer = {col: [] for col in value_cols}
        st.session_state.predicted_labels_history = []
        st.session_state.predicted_label = None
 
        fill_start = max(0, int(jump_target) - window_size)
        for idx in range(fill_start, int(jump_target) + 1):
            r = numeric_df.iloc[idx]
            for col in value_cols:
                st.session_state.buffer[col].append(float(r[col]))
 
        st.session_state.active_value_cols = value_cols
        st.session_state.active_speed = speed
        st.session_state.active_window_size = window_size
        st.rerun()
 
col_a, col_b, col_c = st.columns(3)
with col_a:
    start_btn = st.button("Start", use_container_width=True)
with col_b:
    pause_label = "Resume" if st.session_state.paused else "Pause"
    pause_btn = st.button(pause_label, use_container_width=True)
with col_c:
    stop_btn = st.button("Stop", use_container_width=True)
 
if start_btn:
    st.session_state.running = True
    st.session_state.paused = False
    st.session_state.playback_mode = "playing"
    st.session_state.buffer = {col: [] for col in value_cols}
    st.session_state.predicted_labels_history = []
    st.session_state.predicted_label = None
    st.session_state.active_value_cols = value_cols
    st.session_state.active_speed = speed
    st.session_state.active_window_size = window_size
 
    if data_source == "Recorded Playback":
        selected_row = int(st.session_state.jump_target)
        st.session_state.current_index = selected_row
        st.session_state.prev_jump = selected_row
        st.session_state.jump_target = selected_row
    else:
        # Live mode: start fresh and clear anything already queued
        st.session_state.udp_index = 0
        st.session_state.live_last_row = {}
        try:
            get_udp_receiver(udp_port).drain()
        except Exception:
            pass
 
if pause_btn:
    if st.session_state.playback_mode == "playing":
        st.session_state.playback_mode = "paused"
        st.session_state.paused = True
        st.session_state.running = True
    elif st.session_state.playback_mode == "paused":
        st.session_state.playback_mode = "playing"
        st.session_state.paused = False
        st.session_state.running = True
 
if stop_btn:
    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.playback_mode = "stopped"
    st.session_state.current_index = 0
    st.session_state.prev_jump = 0
    st.session_state.buffer = {}
    st.session_state.predicted_labels_history = []
    st.session_state.predicted_label = None
    st.session_state.active_value_cols = []
    st.session_state.active_speed = speed
    st.session_state.active_window_size = window_size
    st.session_state.udp_index = 0
    st.session_state.live_last_row = {}
 
st.markdown("---")
 
# -----------------------------
# SECTION 4 — LIVE SIGNAL MONITOR
# -----------------------------
st.header("4. Live Signal Monitor")
 
if st.session_state.playback_mode != "stopped" and st.session_state.active_value_cols:
    active_value_cols = st.session_state.active_value_cols
    active_speed = st.session_state.active_speed
    active_window_size = st.session_state.active_window_size
else:
    active_value_cols = value_cols
    active_speed = speed
    active_window_size = window_size
 
play_interval = max(1.0 / max(active_speed, 1), 0.05)
auto_play = st.session_state.playback_mode == "playing"
 
is_live = st.session_state.data_source == "Live UDP Stream"
 
 
def current_values(i):
    """Return a dict of the current per-column values for metric cards,
    sourced from the recorded row or the latest UDP packet."""
    if is_live:
        cur = st.session_state.get("live_last_row", {}) or {}
        out = {}
        for c in active_value_cols:
            v = cur.get(c, None)
            out[c] = float(v) if v is not None else 0.0
        return out
    else:
        row = numeric_df.iloc[i]
        return {c: float(row[c]) for c in active_value_cols if c in numeric_df.columns}
 
 
def render_monitor():
    if is_live:
        i = int(st.session_state.udp_index)
    else:
        i = int(max(0, min(int(st.session_state.current_index), len(df) - 1)))
 
    vals = current_values(i)
 
    if is_live:
        time_val = round(i / 10.0, 2)
        pkts = 0
        try:
            pkts = get_udp_receiver(udp_port).count()
        except Exception:
            pass
        if st.session_state.playback_mode == "playing":
            status_text = f"Live (UDP) — Samples: {i} | Time: {time_val}s | Packets received: {pkts}"
        elif st.session_state.playback_mode == "paused":
            status_text = f"Paused (UDP) — Samples: {i} | Packets received: {pkts}"
        else:
            status_text = f"Stopped (UDP) — Packets received: {pkts}"
        st.markdown(f"**{status_text}**")
    else:
        progress_pct = int((i / max(len(df) - 1, 1)) * 100)
        time_val = round(i / 10.0, 2)
        if st.session_state.playback_mode == "playing":
            status_text = f"Playing — Row {i} / {len(df) - 1} | Time: {time_val}s"
        elif st.session_state.playback_mode == "paused":
            status_text = f"Paused — Row {i} / {len(df) - 1} | Time: {time_val}s"
        else:
            status_text = f"Stopped — Row {i} / {len(df) - 1} | Time: {time_val}s"
        st.markdown(f"**{status_text}**")
        st.progress(progress_pct)
 
    # Surface any prediction error instead of failing silently
    if st.session_state.get("pred_error"):
        st.warning(f"Prediction skipped: {st.session_state.pred_error}")
 
    if st.session_state.model is not None and st.session_state.predicted_label is not None:
        pred = st.session_state.predicted_label
        label_text = label_map.get(pred, str(pred)) if label_map else str(pred)
        color = {"Safe": "green", "Aggressive": "red", "Distracted": "orange"}.get(label_text, "#555")
        st.markdown(
            f"""
            <div style='background-color:{color}22; border-left: 5px solid {color};
            padding: 10px 16px; border-radius: 6px; margin-bottom: 10px;'>
                <span style='font-size:16px; font-weight:bold; color:{color};'>
                    Predicted Class: {label_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
 
    if st.session_state.buffer:
        buf = st.session_state.buffer
        lengths = [len(v) for v in buf.values()] or [0]
        min_len = min(lengths)
        if min_len > 0:
            safe_buf = {k: list(v)[-min_len:] for k, v in buf.items()}
        else:
            safe_buf = {k: [] for k in buf}
        buf_df = pd.DataFrame(safe_buf)
        plot_cols = [c for c in active_value_cols if c in buf]
    else:
        if is_live:
            buf_df = pd.DataFrame({c: [] for c in active_value_cols})
            plot_cols = active_value_cols
        else:
            buf_df = pd.DataFrame({
                col: df[col].iloc[:100].tolist()
                for col in active_value_cols if col in df.columns
            })
            plot_cols = active_value_cols
 
    if len(plot_cols) > 0:
        mcols = st.columns(len(plot_cols))
        for j, col_name in enumerate(plot_cols):
            val = float(vals.get(col_name, 0.0))
            cbuf = st.session_state.buffer.get(col_name, [val]) if st.session_state.buffer else [val]
            prev_val = cbuf[-2] if len(cbuf) > 1 else val
            delta = round(val - prev_val, 4)
            mcols[j].metric(col_name, f"{val:.3f}", delta=f"{delta:+.3f}")
 
    if len(buf_df) > 0 and len(plot_cols) > 0:
        fig, axes = plt.subplots(3, 2, figsize=(14, 9), squeeze=False)
        axes_flat = axes.flatten()
 
        x_vals = list(range(max(0, i - len(buf_df) + 1), max(len(buf_df), i + 1)))
 
        x_plot = x_vals
        x_label = "Sample" if is_live else "Row Index"
        bar_width = 1.0
        x_is_date = False
        x_fallback_msg = None
 
        # Time-column and class-colour handling only apply to recorded playback
        if (not is_live) and time_col != "Row Index" and time_col in df.columns:
            row_idxs = [min(max(v, 0), len(df) - 1) for v in x_vals]
            col_slice = df[time_col].iloc[row_idxs]
 
            numeric_try = pd.to_numeric(col_slice, errors="coerce").values
            datetime_try = pd.to_datetime(col_slice, errors="coerce")
 
            if not np.all(np.isnan(numeric_try)):
                x_plot = numeric_try
                x_label = time_col
            elif not datetime_try.isna().all():
                x_plot = mdates.date2num(datetime_try.to_numpy())
                x_label = time_col
                x_is_date = True
            else:
                x_fallback_msg = (
                    f"Column '{time_col}' isn't numeric or a recognizable date — "
                    f"showing Row Index instead."
                )
 
            if x_label == time_col and len(x_plot) > 1:
                diffs = np.diff(np.asarray(x_plot, dtype=float))
                diffs = diffs[~np.isnan(diffs)]
                if len(diffs) > 0 and np.median(np.abs(diffs)) > 0:
                    bar_width = float(np.median(np.abs(diffs))) * 0.8
 
        if x_fallback_msg:
            st.warning(x_fallback_msg)
 
        class_colors = None
        if (not is_live) and color_by != "None" and color_by in df.columns and len(buf_df) > 0:
            start_idx = max(0, i - len(buf_df) + 1)
            end_idx = min(i + 1, len(df))
            class_series = df[color_by].iloc[start_idx:end_idx].values
            unique_classes = np.unique(class_series)
            cmap = plt.get_cmap("tab10", max(len(unique_classes), 1))
            class_color_map = {cls: cmap(idx) for idx, cls in enumerate(unique_classes)}
            class_colors = [class_color_map.get(c, "steelblue") for c in class_series]
 
            if len(class_colors) > len(buf_df):
                class_colors = class_colors[-len(buf_df):]
            elif len(class_colors) < len(buf_df):
                class_colors = class_colors + ["steelblue"] * (len(buf_df) - len(class_colors))
 
        for idx, col_name in enumerate(plot_cols):
            ax = axes_flat[idx]
            y_vals = buf_df[col_name].values if col_name in buf_df.columns else []
 
            if len(y_vals) == 0:
                ax.set_visible(False)
                continue
 
            # Guard against all-NaN slices in histogram/box
            y_clean = np.asarray(y_vals, dtype=float)
            y_clean = y_clean[~np.isnan(y_clean)]
 
            col_chart_type = per_col_type.get(col_name, graph_type_global)
            base_color = class_colors if class_colors is not None else None
 
            if col_chart_type == "Line":
                if base_color is not None:
                    for k in range(len(x_plot) - 1):
                        ax.plot(
                            x_plot[k:k + 2],
                            y_vals[k:k + 2],
                            color=base_color[k] if k < len(base_color) else "steelblue",
                            linewidth=1.5
                        )
                else:
                    ax.plot(x_plot, y_vals, color="steelblue", linewidth=1.5)
                ax.set_xlabel(x_label, fontsize=8)
                ax.set_ylabel("Value", fontsize=8)
 
            elif col_chart_type == "Scatter":
                colors = base_color if base_color is not None else "steelblue"
                ax.scatter(x_plot, y_vals, c=colors, s=8, alpha=0.7)
                ax.set_xlabel(x_label, fontsize=8)
                ax.set_ylabel("Value", fontsize=8)
 
            elif col_chart_type == "Histogram":
                if len(y_clean) > 0:
                    ax.hist(y_clean, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
                ax.set_xlabel("Value", fontsize=8)
                ax.set_ylabel("Frequency", fontsize=8)
 
            elif col_chart_type == "Area":
                ax.fill_between(x_plot, y_vals, alpha=0.4, color="steelblue")
                ax.plot(x_plot, y_vals, color="steelblue", linewidth=1.0)
                ax.set_xlabel(x_label, fontsize=8)
                ax.set_ylabel("Value", fontsize=8)
 
            elif col_chart_type == "Bar":
                ax.bar(x_plot, y_vals, color="steelblue", alpha=0.7, width=bar_width)
                ax.set_xlabel(x_label, fontsize=8)
                ax.set_ylabel("Value", fontsize=8)
 
            elif col_chart_type == "Box":
                if len(y_clean) > 0:
                    ax.boxplot(
                        y_clean,
                        patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.6),
                        medianprops=dict(color="red", linewidth=2)
                    )
                ax.set_ylabel("Value", fontsize=8)
 
            ax.set_title(col_name, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
 
            if x_is_date and col_chart_type in ("Line", "Scatter", "Area", "Bar"):
                locator = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(30)
                    lbl.set_ha("right")
 
        for idx in range(len(plot_cols), 6):
            axes_flat[idx].set_visible(False)
 
        fig.suptitle("Live Signal Monitor", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
 
 
def _predict_from_row(value_getter):
    """Run model prediction from a callable that returns a column's value.
    Surfaces errors into session_state instead of failing silently."""
    if st.session_state.model is None or len(model_feature_cols) == 0:
        return
    try:
        feats = [float(value_getter(c)) for c in model_feature_cols]
        pred = st.session_state.model.predict(np.array(feats).reshape(1, -1))[0]
        st.session_state.predicted_label = pred
        st.session_state.predicted_labels_history.append(pred)
        st.session_state.pred_error = None
    except Exception as e:
        st.session_state.predicted_label = None
        st.session_state.pred_error = str(e)
 
 
@st.fragment(run_every=play_interval if auto_play else None)
def live_monitor():
    if st.session_state.playback_mode != "playing":
        render_monitor()
        return
 
    cols_to_use = st.session_state.active_value_cols or active_value_cols
    win = st.session_state.active_window_size or active_window_size
 
    # ============ LIVE UDP MODE ============
    if is_live:
        receiver = get_udp_receiver(udp_port)
        new_rows = receiver.drain()
 
        if not st.session_state.buffer:
            st.session_state.buffer = {c: [] for c in cols_to_use}
 
        last_row = None
        for row in new_rows:
            last_row = row
            for c in cols_to_use:
                st.session_state.buffer.setdefault(c, [])
                v = row.get(c, None)
                st.session_state.buffer[c].append(float(v) if v is not None else 0.0)
                if len(st.session_state.buffer[c]) > win:
                    st.session_state.buffer[c] = st.session_state.buffer[c][-win:]
            st.session_state.udp_index += 1
 
        if last_row is not None:
            st.session_state.live_last_row = last_row
            _predict_from_row(lambda c: last_row.get(c, 0.0) if last_row.get(c) is not None else 0.0)
 
        render_monitor()
        return
 
    # ============ RECORDED PLAYBACK MODE ============
    idx = int(st.session_state.current_index)
    idx = max(0, min(idx, len(df) - 1))
 
    if idx >= len(df) - 1:
        # Append the final row, then stop in place (no reset to zero).
        r = numeric_df.iloc[idx]
        if not st.session_state.buffer:
            st.session_state.buffer = {c: [] for c in cols_to_use}
        for c in cols_to_use:
            st.session_state.buffer.setdefault(c, [])
            st.session_state.buffer[c].append(float(r[c]))
            if len(st.session_state.buffer[c]) > win:
                st.session_state.buffer[c] = st.session_state.buffer[c][-win:]
        st.session_state.current_index = idx
        st.session_state.running = False
        st.session_state.paused = True
        st.session_state.playback_mode = "paused"
        render_monitor()
        return
    else:
        r = numeric_df.iloc[idx]
 
        if not st.session_state.buffer:
            st.session_state.buffer = {c: [] for c in cols_to_use}
 
        for c in cols_to_use:
            st.session_state.buffer.setdefault(c, [])
            st.session_state.buffer[c].append(float(r[c]))
            if len(st.session_state.buffer[c]) > win:
                st.session_state.buffer[c] = st.session_state.buffer[c][-win:]
 
        _predict_from_row(lambda c: float(r[c]) if c in numeric_df.columns else 0.0)
 
        st.session_state.current_index = idx + 1
 
    render_monitor()
 
 
live_monitor()
 
st.markdown("---")
 
# -----------------------------
# SECTION 5 — ANNOTATION
# -----------------------------
st.header("5. Annotation")
st.caption("Pause playback/streaming to annotate the current row, then resume.")
 
ann_col1, ann_col2 = st.columns([2, 1])
with ann_col1:
    selected_label = st.selectbox(
        "Label for Current Row",
        ["Safe", "Aggressive", "Distracted", "Fatigued", "Unknown"],
        key="annotation_label"
    )
with ann_col2:
    save_btn = st.button("Save Annotation", use_container_width=True)
 
save_status = st.empty()
 
if save_btn:
    if is_live:
        current_i = int(st.session_state.udp_index)
    else:
        current_i = min(st.session_state.current_index, len(df) - 1)
    existing = [a for a in st.session_state.annotation_data if a["index"] == current_i]
    if existing:
        for a in st.session_state.annotation_data:
            if a["index"] == current_i:
                a["label"] = selected_label
        save_status.success(f"Updated: Row {current_i} to {selected_label}")
    else:
        st.session_state.annotation_data.append({
            "index": current_i,
            "time_s": round(current_i / 10.0, 2),
            "label": selected_label
        })
        save_status.success(f"Saved: Row {current_i} to {selected_label}")
 
if len(st.session_state.annotation_data) > 0:
    st.subheader(f"Saved Annotations ({len(st.session_state.annotation_data)})")
    ann_df = pd.DataFrame(st.session_state.annotation_data).sort_values("index").reset_index(drop=True)
    st.dataframe(ann_df, use_container_width=True)
 
    del_index = st.number_input(
        "Delete Annotation at Row",
        min_value=0, step=1, value=0
    )
    if st.button("Delete Annotation"):
        before = len(st.session_state.annotation_data)
        st.session_state.annotation_data = [
            a for a in st.session_state.annotation_data if a["index"] != del_index
        ]
        after = len(st.session_state.annotation_data)
        if before != after:
            st.success(f"Deleted annotation at row {del_index}.")
        else:
            st.warning(f"No annotation found at row {del_index}.")
        st.rerun()
 
    if st.button("Clear All Annotations"):
        st.session_state.annotation_data = []
        st.rerun()
 
st.markdown("---")
 
# -----------------------------
# SECTION 6 — EXPORT
# -----------------------------
st.header("6. Export")
 
exp_col1, exp_col2 = st.columns(2)
 
with exp_col1:
    if len(st.session_state.annotation_data) > 0:
        ann_df = pd.DataFrame(st.session_state.annotation_data).sort_values("index")
        st.download_button(
            "Download Annotations CSV",
            data=ann_df.to_csv(index=False).encode("utf-8"),
            file_name="annotations.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No annotations saved yet.")
 
with exp_col2:
    if len(st.session_state.annotation_data) > 0:
        export_df = df.copy()
        export_df["corrected_label"] = ""
        for ann in st.session_state.annotation_data:
            idx = ann["index"]
            if 0 <= idx < len(export_df):
                export_df.at[idx, "corrected_label"] = ann["label"]
        st.download_button(
            "Download Full Dataset with Corrected Labels",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="dataset_with_corrected_labels.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Save annotations first to export the full dataset.")
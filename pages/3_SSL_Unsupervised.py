"""Semi-Supervised and Unsupervised Learning.

Five steps, stacked top to bottom: load a large unlabeled dataset, check its
quality, impute missing values, auto label it with a pretrained model (self
training), and cluster it. Results are kept in st.session_state so they
survive Streamlit reruns, and heavy steps are wrapped in try/except.
"""
from pathlib import Path

# pandas/numpy hold and crunch the data, Streamlit renders the page, and
# matplotlib/seaborn draw the charts.
import numpy as np
import pandas as pd
import streamlit as st
from theme import apply_theme

apply_theme()
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn supplies the imputers, base classifiers, clustering methods,
# scalers, and metrics; the top level import only reports versions on errors.
import sklearn
from sklearn.base import clone
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from branding import render_header
from theme import apply_theme

# Checkpoints live next to the app so the path works no matter which
# directory Streamlit was launched from.
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def show_library_error(step, e):
    """Show a failed library call plus the installed versions, because
    functions change between releases (sklearn 1.2 renamed affinity to metric)."""
    st.error(f"{step} failed: {e}")
    st.caption(
        f"pandas {pd.__version__}, numpy {np.__version__}, "
        f"scikit-learn {sklearn.__version__}. If the error mentions an "
        "unexpected argument, check the library's GitHub changelog."
    )


def downcast_numeric(df):
    """Downcast numeric columns to the smallest dtype that fits, replacing
    one column at a time so no second full copy is held in memory."""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def read_csv_with_progress(file, chunksize, downcast=True):
    """Read a CSV one chunk at a time so only one chunk is ever buffered at
    full size, downcasting each chunk right away and showing progress."""
    size = getattr(file, "size", None)
    progress = st.progress(0.0)
    status = st.empty()
    chunks, rows = [], 0
    for chunk in pd.read_csv(file, chunksize=chunksize):
        if downcast:
            chunk = downcast_numeric(chunk)
        chunks.append(chunk)
        rows += len(chunk)
        status.text(f"Read {rows:,} rows...")
        if size:
            progress.progress(min(file.tell() / size, 1.0))
    progress.progress(1.0)
    status.text(f"Finished reading {rows:,} rows.")
    return pd.concat(chunks, ignore_index=True)


def reduce_memory(df):
    """Downcast the frame and report its size before and after in MB."""
    before = df.memory_usage(deep=True).sum() / 1024 ** 2
    df = downcast_numeric(df)
    after = df.memory_usage(deep=True).sum() / 1024 ** 2
    return df, before, after


@st.cache_data(show_spinner=False)
def scale_features(values, scaler_name):
    """Normalize in float32 and cache the result, so the scaler is not
    refit on the whole dataset every time a widget triggers a rerun."""
    scalers = {
        "Standard (z score)": StandardScaler(),
        "Min Max (0 to 1)": MinMaxScaler(),
        "Robust (median and IQR)": RobustScaler(),
    }
    scaled = scalers[scaler_name].fit_transform(values.astype(np.float32))
    return scaled.astype(np.float32)


def elbow_silhouette_scan(X, k_min, k_max):
    """Fit K-Means once per K in the range and record two ways of judging it:
    WCSS (inertia, the elbow curve) and the silhouette score. Silhouette is
    O(n^2), so it is measured on a capped sample."""
    rows = []
    for k in range(int(k_min), int(k_max) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({
            "K": k,
            "WCSS": float(km.inertia_),
            "Silhouette": float(silhouette_score(
                X, km.labels_, sample_size=min(5000, len(X)), random_state=42)),
        })
    return pd.DataFrame(rows)


def save_checkpoint(df):
    """Save the working data to disk, falling back to CSV when parquet
    (pyarrow) is unavailable."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    try:
        path = CHECKPOINT_DIR / "ssl_checkpoint.parquet"
        df.to_parquet(path)
    except Exception:
        path = CHECKPOINT_DIR / "ssl_checkpoint.csv"
        df.to_csv(path, index=False)
    return path


def load_checkpoint():
    """Load the most recent checkpoint, or return None if there is none."""
    parquet = CHECKPOINT_DIR / "ssl_checkpoint.parquet"
    csv = CHECKPOINT_DIR / "ssl_checkpoint.csv"
    if parquet.exists():
        return pd.read_parquet(parquet), parquet
    if csv.exists():
        return pd.read_csv(csv), csv
    return None, None


def run_self_training(pretrained, feature_cols, X_ref, y_ref, unlabeled,
                      threshold, rounds, eval_frac, progress, max_per_round=0):
    """Self training: pseudo label the confident unlabeled rows, add them to
    the training data, retrain, and repeat for several rounds. The labeled
    reference (rows the model never trained on) is split into a training seed
    and a held out set so accuracy is scored honestly; max_per_round caps how
    many pseudo labels are accepted per round (0 = no cap)."""
    can_stratify = y_ref.value_counts().min() >= 2
    seed_X, hold_X, seed_y, hold_y = train_test_split(
        X_ref, y_ref, test_size=eval_frac, random_state=42,
        stratify=y_ref if can_stratify else None,
    )
    seed_y, hold_y = np.asarray(seed_y), np.asarray(hold_y)

    U = unlabeled[feature_cols].reset_index(drop=True)
    remaining = np.ones(len(U), dtype=bool)  # rows not yet pseudo labeled
    pseudo_X, pseudo_y = [], []

    # Round 0: score the pretrained model directly on the held out set.
    history = [{
        "round": 0,
        "accuracy": accuracy_score(hold_y, pretrained.predict(hold_X)),
        "pct_labeled": 0.0,
    }]
    base = clone(pretrained)  # same architecture and hyperparameters
    labeler = current = pretrained

    for r in range(1, rounds + 1):
        # Pseudo label the rows not accepted yet, keeping only the ones at or
        # above the confidence threshold (capped when max_per_round is set).
        rem_idx = np.where(remaining)[0]
        if len(rem_idx) > 0:
            probs = labeler.predict_proba(U.iloc[rem_idx])
            conf = probs.max(axis=1)
            preds = labeler.classes_[probs.argmax(axis=1)]
            accepted = conf >= threshold
            take, take_preds = rem_idx[accepted], preds[accepted]
            if max_per_round and len(take) > max_per_round:
                order = np.argsort(conf[accepted])[::-1][:max_per_round]
                take, take_preds = take[order], take_preds[order]
            if len(take) > 0:
                pseudo_X.append(U.iloc[take])
                pseudo_y.append(take_preds)
                remaining[take] = False

        # Retrain a fresh copy on the seed plus all accepted pseudo labels.
        if pseudo_X:
            train_X = pd.concat([seed_X, pd.concat(pseudo_X)], axis=0)
            train_y = np.concatenate([seed_y, np.concatenate(pseudo_y)])
        else:
            train_X, train_y = seed_X, seed_y
        model_r = clone(base)
        model_r.fit(train_X, train_y)

        pct = (len(U) - remaining.sum()) / len(U) * 100 if len(U) else 0.0
        history.append({
            "round": r,
            "accuracy": accuracy_score(hold_y, model_r.predict(hold_X)),
            "pct_labeled": pct,
        })
        labeler = current = model_r
        progress.progress(r / rounds)

    # Label every unlabeled row with the final model for download.
    probs_all = current.predict_proba(U)
    labeled = unlabeled.copy()
    labeled["pseudo_label"] = current.classes_[probs_all.argmax(axis=1)]
    labeled["confidence"] = probs_all.max(axis=1)

    return {
        "history": pd.DataFrame(history),
        "hold_y": hold_y,
        "final_pred": current.predict(hold_X),
        "labeled": labeled,
        "classes": list(current.classes_),
    }


def _toggle(key):
    """Flip a section's open/closed flag (runs before the script reruns)."""
    st.session_state[key] = not st.session_state.get(key, False)


def reveal_button(key, name):
    """Show/Hide button that keeps its open state in st.session_state,
    because a plain st.button is only True on the run it is clicked."""
    if key not in st.session_state:
        st.session_state[key] = False
    st.button(
        f"Hide {name}" if st.session_state[key] else f"Show {name}",
        key=f"btn_{key}", on_click=_toggle, args=(key,),
    )
    return st.session_state[key]


apply_theme()

render_header(
    title="Semi-Supervised and Unsupervised Learning",
    subtitle_html=(
        "SSL • Clustering • Data Quality Analysis"
        "<br>Advanced Biomechatronics and Locomotion Laboratory"
    ),
)

# Keep every original task in its existing container and order, but present
# the six tasks as compact horizontal tabs instead of one long page.
tabs = st.tabs(
    [
        "1. Large Unlabeled Data",
        "2. Data Quality Analysis",
        "3. Data Imputation",
        "4. Semi-Supervised ML",
        "5. SSL Evaluation",
        "6. Unsupervised Learning",
    ]
)
tab_load = tabs[0]
tab_quality = tabs[1]
tab_impute = tabs[2]
tab_ssl = tabs[3]
tab_eval = tabs[4]
tab_cluster = tabs[5]


# 1. Large Unlabeled Data
with tab_load:
   

    # Chunked reading buffers one slice of the file at a time instead of
    # pulling the whole file into RAM at once.
    chunked = st.checkbox("Read in chunks (recommended for large files)", value=True)
    chunksize = st.number_input("Rows per chunk", min_value=1000, value=50000, step=1000)
    downcast_read = st.checkbox("Downcast dtypes while reading", value=True,
                                disabled=not chunked)
    do_sample = st.checkbox("Sample rows after loading")
    sample_rows = st.number_input("Number of rows to keep", min_value=100,
                                  value=10000, step=100, disabled=not do_sample)

    uploaded = st.file_uploader("Upload unlabeled dataset (CSV)", type=["csv"])

    if uploaded is not None:
        # Crash protection: a bad file is reported instead of killing the app.
        try:
            if chunked:
                df = read_csv_with_progress(uploaded, int(chunksize), downcast=downcast_read)
            else:
                df = pd.read_csv(uploaded)
            if do_sample and len(df) > sample_rows:
                df = df.sample(int(sample_rows), random_state=42).reset_index(drop=True)
                st.info(f"Sampled {sample_rows:,} rows from the file.")
            st.session_state["ssl_raw"] = df
            st.success(f"Loaded dataset. Shape: {df.shape}")
        except Exception as e:
            show_library_error("Reading the file", e)

    if "ssl_raw" in st.session_state:
        df = st.session_state["ssl_raw"]
        st.caption(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns.")
        if reveal_button("show_load_details", "preview and memory tools"):
            st.dataframe(df.head())

            st.subheader("Memory Usage")
            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            st.write(f"Current memory: **{mem_mb:.2f} MB** for {len(df):,} rows.")
            if st.button("Optimize memory (downcast numeric columns)"):
                optimized, before, after = reduce_memory(df)
                st.session_state["ssl_raw"] = optimized
                st.success(
                    f"Reduced memory from {before:.2f} MB to {after:.2f} MB "
                    f"({(1 - after / before) * 100:.1f}% smaller)."
                )

            # Session state survives reruns; the disk checkpoint also covers
            # a full restart or crash.
            st.subheader("Crash Protection and Recovery")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save checkpoint to disk"):
                    try:
                        path = save_checkpoint(st.session_state["ssl_raw"])
                        st.success(f"Checkpoint saved: {path.name}")
                    except Exception as e:
                        st.error(f"Could not save checkpoint: {e}")
            with c2:
                if st.button("Restore last checkpoint"):
                    data, path = load_checkpoint()
                    if data is None:
                        st.warning("No checkpoint found yet.")
                    else:
                        st.session_state["ssl_raw"] = data
                        st.success(f"Restored from {path.name}. Shape: {data.shape}")
    else:
        st.info("Upload an unlabeled dataset to begin.")


# 2. Data Quality Analysis
with tab_quality:
    st.header("2. Data Quality Analysis")

    if "ssl_raw" not in st.session_state:
        st.info("Load a dataset above to analyze its quality.")
    elif reveal_button("show_quality", "data quality results"):
        df = st.session_state["ssl_raw"]

        st.subheader("Missing Values")
        missing = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
        })
        st.dataframe(missing)

        # A signal (column) is flagged incomplete when too many of its
        # values are missing; partly filled rows are counted as well.
        st.subheader("Incomplete Signals")
        threshold_pct = st.slider(
            "Flag a signal as incomplete when missing percent is above", 0, 100, 20)
        incomplete_cols = missing[missing["missing_percent"] > threshold_pct]
        if incomplete_cols.empty:
            st.success("No columns exceed the missing value threshold.")
        else:
            st.warning("These columns look like incomplete signals:")
            st.dataframe(incomplete_cols)

        incomplete_rows = int((df.isna().sum(axis=1) > 0).sum())
        st.write(
            f"Rows with at least one missing value: "
            f"**{incomplete_rows:,}** out of {len(df):,}."
        )

        st.subheader("Summary Statistics")
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            st.warning("No numeric columns to summarize.")
        else:
            st.dataframe(numeric.describe().transpose().round(3))

            st.subheader("Rough Histograms")
            cols = numeric.columns.tolist()
            chosen = st.multiselect(
                "Choose columns to plot", cols, default=cols[: min(4, len(cols))])
            bins = st.slider("Number of bins", 5, 60, 20)
            if chosen:
                n = len(chosen)
                nrows = (n + 1) // 2
                fig, axes = plt.subplots(nrows, 2, figsize=(10, 3 * nrows))
                axes = np.array(axes).reshape(-1)
                for ax, col in zip(axes, chosen):
                    ax.hist(numeric[col].dropna(), bins=bins, color="#B31B1B", alpha=0.8)
                    ax.set_title(col)
                for ax in axes[n:]:  # hide leftover empty subplots
                    ax.axis("off")
                fig.tight_layout()
                st.pyplot(fig)


# 3. Data Imputation
with tab_impute:
    st.header("3. Data Imputation")

    if "ssl_raw" not in st.session_state:
        st.info("Load a dataset above before imputing.")
    elif reveal_button("show_impute", "imputation tools"):
        df = st.session_state["ssl_raw"]
        numeric = df.select_dtypes(include=[np.number])

        if numeric.shape[1] == 0:
            st.warning("No numeric columns available to impute.")
        else:
            # scikit-learn does the actual filling: SimpleImputer covers the
            # statistics based methods, KNNImputer uses the k most similar rows.
            strategy = st.selectbox(
                "Imputation method",
                ["mean", "median", "most_frequent", "constant", "knn"])
            fill_value = None
            n_neighbors = 5
            if strategy == "constant":
                fill_value = st.number_input("Fill value", value=0.0)
            elif strategy == "knn":
                # More neighbors gives smoother fills, fewer follows the
                # local pattern closely.
                n_neighbors = st.slider("Number of neighbors (k)", 1, 15, 5)
                st.caption("Slower on large datasets.")

            st.write("Missing values before imputation:")
            st.write(numeric.isna().sum())

            if st.button("Run Imputation"):
                try:
                    if strategy == "knn":
                        imputer = KNNImputer(n_neighbors=n_neighbors)
                    else:
                        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                    imputed = pd.DataFrame(
                        imputer.fit_transform(numeric), columns=numeric.columns)
                    st.session_state["ssl_clean"] = imputed
                    st.success("Imputation complete.")
                except Exception as e:
                    show_library_error("Imputation", e)

        if "ssl_clean" in st.session_state:
            clean = st.session_state["ssl_clean"]
            st.write("Missing values after imputation:")
            st.write(clean.isna().sum())
            st.dataframe(clean.head())
            st.download_button(
                "Download imputed dataset",
                data=clean.to_csv(index=False).encode("utf-8"),
                file_name="imputed_unlabeled_data.csv", mime="text/csv")


# 4. Semi-Supervised ML
with tab_ssl:
    st.header("4. Semi-Supervised ML")

    if "ssl_clean" not in st.session_state:
        st.info("Impute the unlabeled data above before running SSL.")
    elif reveal_button("show_ssl", "SSL tools"):
        clean = st.session_state["ssl_clean"]

        # The pretrained model and its labeled reference come from Tab 2, or
        # from a labeled dataset uploaded here and split into train/test.
        st.subheader("Pretrained Model and Labeled Reference")
        source = st.radio(
            "Pretrained model source",
            ["Transfer the model trained in Tab 2",
             "Upload a labeled dataset and train a base model here"])

        ref = None
        if source == "Transfer the model trained in Tab 2":
            trained = st.session_state.get("trained")
            if trained is None:
                st.info("Train a classification model in Tab 2 first.")
            elif trained.get("problem_type") != "Classification":
                st.warning("The Tab 2 model is regression. SSL needs a "
                           "classification model.")
            else:
                st.success(f"Using the Tab 2 model: **{trained['model_name']}**")
                ref = {
                    "model": trained["model"],
                    "model_name": trained["model_name"],
                    "feature_cols": trained["feature_cols"],
                    "X_ref": trained["X_test"][trained["feature_cols"]],
                    "y_ref": pd.Series(trained["y_test"]),
                }
        else:
            labeled_file = st.file_uploader(
                "Upload labeled dataset (CSV)", type=["csv"], key="ssl_labeled_file")
            if labeled_file is None:
                st.info("The labeled file needs the same feature columns as "
                        "the unlabeled data plus a label column.")
            else:
                labeled_df = None
                try:
                    labeled_df = pd.read_csv(labeled_file)
                except Exception as e:
                    show_library_error("Reading the labeled file", e)

                if labeled_df is not None:
                    cols = labeled_df.columns.tolist()
                    target_col = st.selectbox("Label column", cols, index=len(cols) - 1)
                    # Only columns present in BOTH files can be features,
                    # since the model must predict on the unlabeled rows.
                    shared = [
                        c for c in cols
                        if c != target_col and c in clean.columns
                        and pd.api.types.is_numeric_dtype(labeled_df[c])
                    ]
                    if not shared:
                        st.error("No numeric feature columns match the unlabeled "
                                 "data. Both files need the same column names.")
                    else:
                        st.caption(f"Shared feature columns: {shared}")

                        # All three base models provide predict_proba, which
                        # self training needs to measure confidence.
                        base_name = st.selectbox(
                            "Base model",
                            ["Random Forest", "Logistic Regression", "KNN"])
                        if base_name == "Random Forest":
                            n_estimators = st.slider(
                                "Number of trees (n_estimators)", 10, 300, 100, 10)
                            max_depth = st.slider(
                                "Max tree depth (0 = unlimited)", 0, 30, 0)
                        elif base_name == "Logistic Regression":
                            C = st.slider(
                                "Regularization strength C", 0.01, 10.0, 1.0, 0.01)
                            max_iter = st.slider("Max iterations", 100, 2000, 500, 100)
                        else:
                            knn_k = st.slider("Number of neighbors (k)", 1, 25, 5)

                        test_size = st.slider("Testing set share", 0.2, 0.5, 0.3, 0.05)

                        if st.button("Train base model on the labeled data"):
                            try:
                                # The base model learns from complete rows only.
                                rows = labeled_df[shared + [target_col]].dropna()
                                X, y = rows[shared], rows[target_col]
                                strat = y if y.value_counts().min() >= 2 else None
                                X_tr, X_te, y_tr, y_te = train_test_split(
                                    X, y, test_size=test_size,
                                    random_state=42, stratify=strat)
                                if base_name == "Random Forest":
                                    base_model = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth or None, random_state=42)
                                elif base_name == "Logistic Regression":
                                    base_model = LogisticRegression(C=C, max_iter=max_iter)
                                else:
                                    base_model = KNeighborsClassifier(n_neighbors=knn_k)
                                base_model.fit(X_tr, y_tr)
                                acc = accuracy_score(y_te, base_model.predict(X_te))
                                st.session_state["ssl_base"] = {
                                    "model": base_model,
                                    "model_name": base_name,
                                    "feature_cols": shared,
                                    "X_ref": X_te,
                                    "y_ref": pd.Series(y_te),
                                    "test_acc": acc,
                                }
                            except Exception as e:
                                show_library_error("Training the base model", e)

                        base = st.session_state.get("ssl_base")
                        if base is not None:
                            st.success(
                                f"Base model trained: **{base['model_name']}** "
                                f"(accuracy {base['test_acc']:.3f} on the testing set).")
                            ref = base

        if ref is not None:
            model = ref["model"]
            feature_cols = ref["feature_cols"]
            missing_feats = [c for c in feature_cols if c not in clean.columns]

            if not hasattr(model, "predict_proba"):
                st.error("This model has no predict_proba, which self training "
                         "needs. Use Random Forest, Logistic Regression, or KNN.")
            elif missing_feats:
                st.error("The unlabeled data is missing feature columns the "
                         f"model expects: {missing_feats}")
            else:
                # Hyperparameters: how picky the loop is, how long it runs,
                # the honest scoring share, and how fast pseudo labels grow.
                st.subheader("Self Training Settings")
                threshold = st.slider(
                    "Confidence threshold for pseudo labels", 0.50, 0.99, 0.90, 0.01)
                rounds = st.slider("Number of training rounds", 1, 10, 3)
                eval_frac = st.slider(
                    "Share of labeled data held out for scoring", 0.2, 0.5, 0.3, 0.05)
                max_per_round = st.number_input(
                    "Max pseudo labels per round (0 = no limit)",
                    min_value=0, value=0, step=100)

                if st.button("Run Semi-Supervised Training"):
                    try:
                        progress = st.progress(0.0)
                        with st.spinner("Auto labeling and retraining..."):
                            result = run_self_training(
                                pretrained=model, feature_cols=feature_cols,
                                X_ref=ref["X_ref"], y_ref=ref["y_ref"],
                                unlabeled=clean, threshold=threshold,
                                rounds=rounds, eval_frac=eval_frac,
                                progress=progress,
                                max_per_round=int(max_per_round))
                        st.session_state["ssl_result"] = result
                        st.success("Training complete. See Tab 5 for the "
                                   "evaluation results.")
                    except Exception as e:
                        show_library_error("Semi supervised training", e)


# 5. SSL Evaluation
with tab_eval:
    st.header("5. SSL Evaluation")

    # Results are rendered from session_state so they survive reruns and stay
    # visible while the training controls live in the previous tab.
    if "ssl_result" not in st.session_state:
        st.info("Run semi-supervised training in Tab 4 before evaluating.")
    else:
        res = st.session_state["ssl_result"]
        history = res["history"]

        st.subheader("Training Progress")
        final_pct = history["pct_labeled"].iloc[-1]
        st.metric("Unlabeled data auto labeled", f"{final_pct:.1f}%")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(history["round"], history["accuracy"], marker="o", color="#B31B1B")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy on held out set")
        ax1.set_title("Accuracy across rounds")
        ax2.plot(history["round"], history["pct_labeled"], marker="s", color="#1B5EB3")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Percent of data labeled")
        ax2.set_title("Auto labeling progress")
        fig.tight_layout()
        st.pyplot(fig)

        # Standard metrics, all measured on the held out labeled set.
        st.subheader("Evaluation Metrics")
        y_true, y_pred = res["hold_y"], res["final_pred"]
        st.write(f"**Accuracy:** {accuracy_score(y_true, y_pred):.3f}")
        st.write(
            f"**Precision (weighted):** "
            f"{precision_score(y_true, y_pred, average='weighted', zero_division=0):.3f}")
        st.write(
            f"**Recall (weighted):** "
            f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):.3f}")
        st.write(
            f"**F1 Score (weighted):** "
            f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.3f}")

        st.subheader("Per Class Report")
        report = classification_report(y_true, y_pred, output_dict=True,
                                       zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(3))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=res["classes"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=res["classes"], yticklabels=res["classes"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        st.subheader("Auto Labeled Data")
        st.dataframe(res["labeled"].head())
        st.download_button(
            "Download auto labeled dataset",
            data=res["labeled"].to_csv(index=False).encode("utf-8"),
            file_name="auto_labeled_data.csv", mime="text/csv")


# 6. Unsupervised Learning
with tab_cluster:
    st.header("6. Unsupervised Learning")

    if "ssl_clean" not in st.session_state:
        st.info("Impute the unlabeled data above before clustering.")
    elif reveal_button("show_cluster", "clustering tools"):
        data = st.session_state["ssl_clean"].copy()
        feature_cols = data.columns.tolist()

        # Choosing K first: the elbow curve shows where extra clusters stop
        # cutting WCSS much, the silhouette peak shows where they are best
        # separated. Both are run on z score scaled features.
        st.subheader("Find the Optimal Number of Clusters")
        c1, c2 = st.columns(2)
        k_min = c1.number_input("Smallest K to test", min_value=2, max_value=20,
                                value=3, step=1)
        k_max = c2.number_input("Largest K to test", min_value=3, max_value=20,
                                value=8, step=1)
        st.caption("Uses K-Means on standard (z score) scaled features.")

        if st.button("Run elbow and silhouette analysis"):
            if k_max <= k_min:
                st.warning("The largest K must be greater than the smallest K.")
            else:
                try:
                    with st.spinner("Fitting K-Means for each K..."):
                        st.session_state["ssl_k_scan"] = elbow_silhouette_scan(
                            scale_features(data[feature_cols], "Standard (z score)"),
                            k_min, k_max)
                except Exception as e:
                    show_library_error("Elbow and silhouette analysis", e)

        if "ssl_k_scan" in st.session_state:
            scan = st.session_state["ssl_k_scan"]

            # One row per K, with both scores side by side for comparison.
            st.write("**K-Means, Elbow (WCSS) and Silhouette values**")
            st.dataframe(
                scan.style.format({"WCSS": "{:,.2f}", "Silhouette": "{:.4f}"}),
                hide_index=True, use_container_width=True)

            best = scan.loc[scan["Silhouette"].idxmax()]

            p1, p2 = st.columns(2)
            with p1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(scan["K"], scan["WCSS"], marker="o", color="#1B5EB3")
                ax.set_xlabel("Number of clusters")
                ax.set_ylabel("WCSS")
                ax.set_title("Elbow Plot")
                fig.tight_layout()
                st.pyplot(fig)
            with p2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(scan["K"], scan["Silhouette"], marker="o", color="#B31B1B")
                # Mark the peak so the best K is easy to spot.
                ax.plot(best["K"], best["Silhouette"], marker="o", markersize=10,
                        color="#444444")
                ax.set_xlabel("Number of clusters")
                ax.set_ylabel("Silhouette score")
                ax.set_title("Silhouette Plot")
                fig.tight_layout()
                st.pyplot(fig)

            st.success(
                f"Best Silhouette result: K={int(best['K'])} with score "
                f"{best['Silhouette']:.4f}.")

        method = st.selectbox(
            "Clustering method",
            ["K-Means", "DBSCAN", "Hierarchical (Agglomerative)"])

        # Hierarchical clustering builds an n x n distance matrix, so large
        # data is sampled for that method only; K-Means and DBSCAN scale fine.
        n_total = len(data)
        if method == "Hierarchical (Agglomerative)" and n_total > 5000:
            max_rows = st.slider("Max rows for hierarchical clustering",
                                 1000, min(20000, n_total), 5000, 1000)
            if n_total > max_rows:
                data = data.sample(max_rows, random_state=42).reset_index(drop=True)
                st.info(f"Using a random sample of {max_rows:,} of {n_total:,} "
                        "rows to limit memory.")

        # Normalization puts every column on a comparable scale so none
        # dominates the distances; scale_features caches the float32 result.
        scaler_name = st.selectbox(
            "Normalization method",
            ["Standard (z score)", "Min Max (0 to 1)", "Robust (median and IQR)"])
        X_scaled = scale_features(data[feature_cols], scaler_name)

        labels = None
        try:
            if method == "K-Means":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                # k-means++ spreads the starting centers, n_init keeps the
                # best of several restarts, max_iter bounds each run.
                with st.expander("More K-Means hyperparameters"):
                    init = st.selectbox("Center initialization (init)",
                                        ["k-means++", "random"])
                    n_init = st.slider("Restarts (n_init)", 1, 20, 10)
                    max_iter = st.slider("Max iterations per run (max_iter)",
                                         100, 500, 300, 50)
                km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                            max_iter=max_iter, random_state=42)
                labels = km.fit_predict(X_scaled)

            elif method == "DBSCAN":
                # DBSCAN clusters by density: eps sets the neighborhood size
                # and min_samples how many points make a dense region.
                eps = st.slider("eps (neighborhood size)", 0.1, 5.0, 0.5, 0.1)
                min_samples = st.slider("min_samples", 2, 20, 5)
                metric = st.selectbox("Distance metric",
                                      ["euclidean", "manhattan", "cosine"])
                db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = db.fit_predict(X_scaled)
                st.caption(f"Points labeled as noise (cluster = -1): "
                           f"{int((labels == -1).sum()):,}")

            else:  # Hierarchical
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                linkage = st.selectbox("Linkage",
                                       ["ward", "complete", "average", "single"])
                if linkage == "ward":  # ward only supports euclidean
                    metric = "euclidean"
                    st.caption("Ward linkage uses the euclidean metric.")
                else:
                    metric = st.selectbox("Distance metric",
                                          ["euclidean", "manhattan", "cosine"])
                agg = AgglomerativeClustering(n_clusters=n_clusters,
                                              linkage=linkage, metric=metric)
                labels = agg.fit_predict(X_scaled)
        except Exception as e:
            show_library_error("Clustering with these settings", e)
            labels = None

        if labels is not None:
            data["cluster"] = labels

            st.subheader("Cluster Sizes")
            st.write(pd.Series(labels).value_counts().sort_index())

            # Silhouette needs at least two real clusters (DBSCAN noise is
            # ignored) and is O(n^2), so it is measured on a capped sample.
            real = labels[labels != -1] if method == "DBSCAN" else labels
            mask = labels != -1 if method == "DBSCAN" else np.ones(len(labels), bool)
            if len(set(real)) >= 2 and mask.sum() > len(set(real)):
                try:
                    score = silhouette_score(
                        X_scaled[mask], labels[mask],
                        sample_size=min(5000, int(mask.sum())), random_state=42)
                    st.metric("Silhouette score", f"{score:.3f}")
                    st.caption("Closer to 1 means tighter, well separated clusters.")
                except Exception as e:
                    st.caption(f"Silhouette score unavailable: {e}")

            # Project to 2D with PCA when there are more than two features.
            st.subheader("Cluster Visualization")
            if len(feature_cols) > 2:
                coords = PCA(n_components=2).fit_transform(X_scaled)
                x_label, y_label = "PC1", "PC2"
            else:
                coords = X_scaled
                x_label, y_label = feature_cols[0], feature_cols[-1]

            fig, ax = plt.subplots()
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                                 cmap="tab10", alpha=0.7)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{method} clusters")
            legend = ax.legend(*scatter.legend_elements(), title="Cluster")
            ax.add_artist(legend)
            st.pyplot(fig)

            st.download_button(
                "Download clustered dataset",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="clustered_unlabeled_data.csv", mime="text/csv")

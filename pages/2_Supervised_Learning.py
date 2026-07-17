import streamlit as st
from theme import apply_theme

apply_theme()
import pandas as pd
import numpy as np
import ast
import time
import os
import joblib




from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Optional TensorFlow / Keras support (CNN / RNN / LSTM — 3.2.1 Deep Learning)
# ---------------------------------------------------------------------------
try:
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import (
        Dense, Dropout, SimpleRNN, LSTM, Conv1D, GlobalMaxPooling1D, Reshape
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import Callback
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_hyperparams(param_str: str) -> dict:
    """Parse a 'key=value, key2=value2' string into a dict (kept from original app)."""
    params = {}
    for item in param_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            params[key] = ast.literal_eval(value)
        except Exception:
            params[key] = value
    return params


def confusion_derived_rates(cm: np.ndarray) -> dict:
    """Macro-averaged TPR, FPR, FNR, TNR from an n-class confusion matrix (3.2.4)."""
    n = cm.shape[0]
    total = cm.sum()
    tprs, fprs, fnrs, tnrs = [], [], [], []
    for i in range(n):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP
        tprs.append(TP / (TP + FN) if (TP + FN) > 0 else 0.0)
        fnrs.append(FN / (TP + FN) if (TP + FN) > 0 else 0.0)
        fprs.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
        tnrs.append(TN / (FP + TN) if (FP + TN) > 0 else 0.0)
    return {
        "TPR": float(np.mean(tprs)),
        "FPR": float(np.mean(fprs)),
        "FNR": float(np.mean(fnrs)),
        "TNR": float(np.mean(tnrs)),
    }


def plot_roc_curve(y_test, y_score, classes):
    """ROC curve(s) + AUC, binary or multiclass one-vs-rest (3.2.4)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    auc_scores = {}
    n_classes = len(classes)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#B31B1B", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        auc_scores["overall"] = roc_auc
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[str(c)] = roc_auc
            ax.plot(fpr, tpr, lw=1.5, label=f"Class {c} (AUC = {roc_auc:.3f})")
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        micro_auc = auc(fpr_micro, tpr_micro)
        auc_scores["micro-average"] = micro_auc
        ax.plot(fpr_micro, tpr_micro, "k--", lw=2, label=f"Micro-average (AUC = {micro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig, auc_scores


def plot_pr_curve(y_test, y_score, classes):
    """Precision-Recall curve(s), mirrors the MATLAB Classification Learner PR plot."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    n_classes = len(classes)
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1], pos_label=classes[1])
        ap = average_precision_score(y_test == classes[1], y_score[:, 1])
        ax.plot(recall, precision, color="#B31B1B", lw=2, label=f"PR (AP = {ap:.3f})")
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        for i, c in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
            ax.plot(recall, precision, lw=1.5, label=f"Class {c} (AP = {ap:.3f})")
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_prediction_visualization(y_test, y_pred, classes):
    """Prediction visualization plot required by 3.2.4: predicted vs actual class counts
    plus a per-sample correctness scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Predicted vs actual counts per class
    true_counts = pd.Series(y_test).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(classes, fill_value=0)
    x = np.arange(len(classes))
    width = 0.35
    axes[0].bar(x - width / 2, true_counts.values, width, label="Actual", color="#4C72B0")
    axes[0].bar(x + width / 2, pred_counts.values, width, label="Predicted", color="#B31B1B")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Predicted vs Actual (class counts)")
    axes[0].legend()

    # Per-sample correctness scatter (first up to 200 test samples)
    n_show = min(200, len(y_test))
    idx = np.arange(n_show)
    y_test_arr = np.asarray(y_test)[:n_show]
    y_pred_arr = np.asarray(y_pred)[:n_show]
    correct = y_test_arr == y_pred_arr
    axes[1].scatter(idx[correct], idx[correct] * 0 + 1, color="#4C72B0", s=14, label="Correct")
    axes[1].scatter(idx[~correct], idx[~correct] * 0, color="#B31B1B", s=14, label="Incorrect")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Incorrect", "Correct"])
    axes[1].set_xlabel("Test sample index")
    axes[1].set_title("Prediction Correctness (sample view)")
    axes[1].legend(loc="center right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Keras (deep learning) helpers — used only when TensorFlow is available
# ---------------------------------------------------------------------------
if TF_AVAILABLE:
    class TimeoutCallback(Callback):
        """Stops training once a wall-clock timeout threshold is exceeded (3.2.2)."""
        def __init__(self, timeout_seconds):
            super().__init__()
            self.timeout_seconds = timeout_seconds
            self.start_time = None

        def on_train_begin(self, logs=None):
            self.start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            if self.timeout_seconds and (time.time() - self.start_time) > self.timeout_seconds:
                self.model.stop_training = True

    class LossThresholdCallback(Callback):
        """Stops training once the training loss falls at/below a target threshold (3.2.2)."""
        def __init__(self, threshold):
            super().__init__()
            self.threshold = threshold

        def on_epoch_end(self, epoch, logs=None):
            loss_val = logs.get("loss") if logs else None
            if self.threshold is not None and loss_val is not None and loss_val <= self.threshold:
                self.model.stop_training = True

    class StProgressCallback(Callback):
        """Live training-progress indicator (3.2.3 Visualization)."""
        def __init__(self, epochs, progress_bar, status_text):
            super().__init__()
            self.epochs = epochs
            self.progress_bar = progress_bar
            self.status_text = status_text

        def on_epoch_end(self, epoch, logs=None):
            pct = min(1.0, (epoch + 1) / max(self.epochs, 1))
            self.progress_bar.progress(pct)
            loss_val = logs.get("loss", float("nan"))
            acc_val = logs.get("accuracy", logs.get("acc", float("nan")))
            val_loss = logs.get("val_loss")
            msg = f"Epoch {epoch + 1}/{self.epochs} — loss: {loss_val:.4f}, accuracy: {acc_val:.4f}"
            if val_loss is not None:
                msg += f", val_loss: {val_loss:.4f}"
            self.status_text.text(msg)

    def build_keras_model(kind, input_dim, num_classes, learning_rate):
        model = Sequential()
        if kind == "CNN (optional)":
            model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
            model.add(Conv1D(32, kernel_size=3, activation="relu", padding="same"))
            model.add(Conv1D(64, kernel_size=3, activation="relu", padding="same"))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(32, activation="relu"))
        elif kind == "RNN":
            model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
            model.add(SimpleRNN(32))
            model.add(Dense(32, activation="relu"))
        elif kind == "LSTM":
            model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
            model.add(LSTM(32))
            model.add(Dense(32, activation="relu"))
        else:  # Fallback dense body
            model.add(Dense(64, activation="relu", input_shape=(input_dim,)))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation="relu"))

        if num_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(num_classes, activation="softmax"))
            loss = "categorical_crossentropy"

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"])
        return model, loss


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("trained_models", {})   # run_label -> {model, is_keras, meta}
st.session_state.setdefault("training_runs", [])     # list of metric dicts for comparison table
st.session_state.setdefault("active_run", None)      # currently selected run label for export/testing
st.session_state.setdefault("loaded_pretrained", None)  # externally loaded model info


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.image("assets/icaav_logo.png", width=100)

with col2:
    st.markdown(
        """
        <div class="icaav-page-title">
            Tab 2 — Supervised Machine Learning
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
    st.image("assets/carleton_logo.png", width=100)

st.markdown("---")



# ===========================================================================
# 1. Load Engineered Dataset
# ===========================================================================
st.header("1. Load Engineered Dataset")
uploaded = st.file_uploader("Upload engineered dataset (CSV)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    target_col = st.selectbox("Select label/target column", df.columns)
    feature_cols = st.multiselect(
        "Select feature columns",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col]
    )

    st.subheader("Model Type")
    model_family = st.radio(
        "Choose the model type",
        ["Classical ML", "Deep Learning"],
        horizontal=True,
        help=(
            "Classical ML offers Classification and Regression algorithms "
            "(SVM, KNN, Decision Tree, Random Forest, Logistic Regression, Naive Bayes, etc.). "
            "Deep Learning offers neural network architectures (FNN, CNN, RNN, LSTM)."
        )
    )

    problem_type = st.selectbox("Problem type", ["Classification", "Regression"], index=1)

    X = df[feature_cols]
    y = df[target_col]

    if X.select_dtypes(include=[np.number]).shape[1] != X.shape[1]:
        st.error("Selected features must all be numeric. Remove text/categorical columns or encode them before training.")
    elif problem_type == "Classification" and y.nunique() < 2:
        st.error("Target column must contain at least two classes. Select a different target column or provide a dataset with multiple classes.")
    elif problem_type == "Regression" and not np.issubdtype(y.dtype, np.number):
        st.error("Regression target must be numeric. Select a continuous target column like price.")
    else:
        if problem_type == "Classification":
            if y.dtype.kind in "bifc" and y.nunique() > 20:
                st.warning(
                    "The selected target appears to be continuous. "
                    "Classification works best with categorical labels."
                )

        # -------------------------------------------------------------
        # 3.2.2 Training Configuration — Train / Validation / Test split
        # -------------------------------------------------------------
        st.header("2. Training Configuration")
        st.subheader("2.1 Train / Validation / Test Split")

        split_c1, split_c2 = st.columns(2)
        with split_c1:
            test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        with split_c2:
            val_size_of_remaining = st.slider(
                "Validation size (% of remaining after test split)", 0, 40, 15
            ) / 100

        random_state = 42

        try:
            if problem_type == "Classification":
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                if val_size_of_remaining > 0:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_trainval, y_trainval, test_size=val_size_of_remaining,
                        random_state=random_state, stratify=y_trainval
                    )
                else:
                    X_train, y_train = X_trainval, y_trainval
                    X_val, y_val = None, None
            else:
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                if val_size_of_remaining > 0:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_trainval, y_trainval, test_size=val_size_of_remaining,
                        random_state=random_state
                    )
                else:
                    X_train, y_train = X_trainval, y_trainval
                    X_val, y_val = None, None
        except ValueError as e:
            if problem_type == "Classification":
                st.error("Please select appropriate target variable.")
            else:
                st.error("Unable to split the dataset. Check the target and feature selection.")
            st.write("Detailed error:", e)
            X_train = X_test = y_train = y_test = None
            X_val = y_val = None

        if X_train is not None:
            shape_msg = f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
            if X_val is not None:
                shape_msg += f", Validation shape: {X_val.shape}"
            st.write(shape_msg)

            # -------------------------------------------------------------
            # 3.2.2 Cross-Validation (Classical ML) / Deep Learning Training Controls
            # -------------------------------------------------------------
            if model_family == "Classical ML":
                st.subheader("2.2 Cross-Validation")
                cv_col1, cv_col2 = st.columns(2)
                with cv_col1:
                    use_cv = st.checkbox("Enable cross-validation", value=False)
                with cv_col2:
                    cv_folds = st.slider("Number of folds (k)", 2, 10, 5, disabled=not use_cv)
            else:
                use_cv = False
                cv_folds = 5
                st.subheader("2.2 Deep Learning Training Controls")
                st.caption("Epoch limit, loss threshold, timeout, batch size and learning rate for the Deep Learning model configured below.")
                dl_c1, dl_c2, dl_c3 = st.columns(3)
                with dl_c1:
                    epoch_limit = st.number_input("Epoch limit", 1, 2000, 50)
                    batch_size = st.number_input("Batch size", 1, 1024, 32)
                with dl_c2:
                    learning_rate_dl = st.number_input(
                        "Learning rate", min_value=0.00001, max_value=1.0, value=0.001,
                        step=0.0001, format="%.5f"
                    )
                    loss_threshold = st.number_input(
                        "Loss threshold (stop when training loss ≤ this value)",
                        min_value=0.0, max_value=10.0, value=0.0, step=0.01,
                        help="Set to 0 to disable early stopping on loss threshold."
                    )
                with dl_c3:
                    timeout_threshold = st.number_input(
                        "Timeout threshold (seconds, 0 = no limit)", 0, 36000, 0
                    )
                    early_stopping_patience = st.number_input(
                        "Early-stopping patience (epochs, 0 = disabled)", 0, 200, 10
                    )

            # ===========================================================
            # 3.2.1 Model Selection & Training
            # ===========================================================
            if model_family == "Classical ML":
                st.header("3. Classical ML — Model Selection & Training")
            else:
                st.header("3. Deep Learning — Model Selection & Training")
                if problem_type == "Regression":
                    st.info("Deep Learning is optimized for classification. Regression support is limited.")

            model_name = None
            model = None
            param_grid = None
            is_keras = False

            # -----------------------------------------------------------
            # 3.2.1 Classical ML Models (existing configuration, kept)
            # -----------------------------------------------------------
            if model_family == "Classical ML":
                if problem_type == "Classification":
                    model_name = st.selectbox(
                        "Choose model",
                        [
                            "Logistic Regression", "SVM", "Random Forest", "KNN",
                            "Decision Tree", "Naive Bayes", "Gradient Boosting",
                            "AdaBoost", "Extra Trees"
                        ]
                    )
                else:
                    model_name = st.selectbox(
                        "Choose model",
                        [
                            "Linear Regression", "Random Forest", "KNN",
                            "Decision Tree", "SVR", "Gradient Boosting",
                            "AdaBoost", "Extra Trees"
                        ]
                    )

                # Optional UI to set common hyperparameters per selected model
                ui_params = {}
                with st.expander("Set hyperparameters for selected model (optional)"):
                    if model_name in ["Random Forest", "Extra Trees"]:
                        n_estimators_ui = st.slider("n_estimators", 10, 500, 100)
                        max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20], index=0)
                        max_depth_ui = None if max_depth_opt == "None" else int(max_depth_opt)
                        min_samples_split_ui = st.number_input("min_samples_split", 2, 100, 2)
                        ui_params = {
                            "n_estimators": n_estimators_ui,
                            "max_depth": max_depth_ui,
                            "min_samples_split": int(min_samples_split_ui),
                        }
                    elif model_name == "KNN":
                        n_neighbors_ui = st.slider("n_neighbors", 1, 50, 5)
                        weights_ui = st.selectbox("weights", ["uniform", "distance"])
                        metric_ui = st.selectbox("metric", ["minkowski", "euclidean", "manhattan"])
                        ui_params = {"n_neighbors": n_neighbors_ui, "weights": weights_ui, "metric": metric_ui}
                    elif model_name == "Decision Tree":
                        criterion_ui = st.selectbox("criterion", ["gini", "entropy", "log_loss"], index=0)
                        max_depth_dt_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20], index=0)
                        max_depth_dt = None if max_depth_dt_opt == "None" else int(max_depth_dt_opt)
                        min_samples_leaf_ui = st.number_input("min_samples_leaf", 1, 100, 1)
                        ui_params = {"criterion": criterion_ui, "max_depth": max_depth_dt, "min_samples_leaf": int(min_samples_leaf_ui)}
                    elif model_name == "SVR" or model_name == "SVM":
                        C_ui = st.number_input("C", 0.01, 100.0, 1.0)
                        kernel_ui = st.selectbox("kernel", ["rbf", "linear", "poly"])
                        epsilon_ui = st.number_input("epsilon", 0.0, 1.0, 0.1)
                        ui_params = {"C": C_ui, "kernel": kernel_ui, "epsilon": epsilon_ui}
                    elif model_name == "Gradient Boosting":
                        learning_rate_ui = st.number_input("learning_rate", 0.001, 1.0, 0.1)
                        n_estimators_gb = st.slider("n_estimators", 10, 500, 100)
                        subsample_ui = st.slider("subsample", 0.1, 1.0, 1.0)
                        ui_params = {"learning_rate": learning_rate_ui, "n_estimators": n_estimators_gb, "subsample": subsample_ui}
                    elif model_name == "AdaBoost":
                        estimator_choice = st.selectbox("estimator", ["Default", "Decision Tree"], index=0)
                        n_estimators_ab = st.slider("n_estimators", 10, 500, 50)
                        learning_rate_ab = st.number_input("learning_rate", 0.01, 1.0, 1.0)
                        ui_params = {"estimator": estimator_choice, "n_estimators": n_estimators_ab, "learning_rate": learning_rate_ab}

                tuning_mode = st.selectbox("Training mode", ["Manual", "Auto-tune hyperparameters"])
                auto_tune = tuning_mode == "Auto-tune hyperparameters"

                manual_params_str = st.text_input(
                    "Manual hyperparameters",
                    value="",
                    help="Enter comma-separated hyperparameters like C=1.0, max_depth=5. Leave blank to use default/manual widget values."
                )
                manual_params = parse_hyperparams(manual_params_str)

                if model_name == "Logistic Regression":
                    if auto_tune:
                        model = LogisticRegression(max_iter=1000)
                        param_grid = {"C": [0.01, 0.1, 1, 10]}
                    else:
                        if manual_params:
                            model = LogisticRegression(max_iter=1000, **manual_params)
                        else:
                            C = st.number_input("C (regularization strength)", 0.01, 10.0, 1.0)
                            model = LogisticRegression(max_iter=1000, C=C)
                elif model_name == "SVM":
                    if auto_tune:
                        model = SVC(probability=True) if problem_type == "Classification" else SVR()
                        param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}
                    else:
                        if manual_params:
                            manual_params.setdefault("probability", True)
                            model = SVC(**manual_params) if problem_type == "Classification" else SVR(**{k: v for k, v in manual_params.items() if k != "probability"})
                        else:
                            C = st.number_input("C (regularization strength)", 0.01, 10.0, 1.0)
                            if problem_type == "Classification":
                                model = SVC(C=C, probability=True)
                            else:
                                model = SVR(C=C)
                elif model_name == "Random Forest":
                    if auto_tune:
                        model = RandomForestClassifier(random_state=42) if problem_type == "Classification" else RandomForestRegressor(random_state=42)
                        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
                    else:
                        if manual_params:
                            model = RandomForestClassifier(random_state=42, **manual_params) if problem_type == "Classification" else RandomForestRegressor(random_state=42, **manual_params)
                        else:
                            if ui_params:
                                rf_params = {k: v for k, v in ui_params.items() if v is not None}
                                model = RandomForestClassifier(random_state=42, **rf_params) if problem_type == "Classification" else RandomForestRegressor(random_state=42, **rf_params)
                            else:
                                n_estimators = st.slider("n_estimators", 10, 300, 100)
                                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42) if problem_type == "Classification" else RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                elif model_name == "KNN":
                    if auto_tune:
                        model = KNeighborsClassifier() if problem_type == "Classification" else KNeighborsRegressor()
                        param_grid = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
                    else:
                        if manual_params:
                            model = KNeighborsClassifier(**manual_params) if problem_type == "Classification" else KNeighborsRegressor(**manual_params)
                        else:
                            if ui_params:
                                knn_params = ui_params.copy()
                                model = KNeighborsClassifier(**knn_params) if problem_type == "Classification" else KNeighborsRegressor(**knn_params)
                            else:
                                n_neighbors = st.slider("n_neighbors", 1, 30, 5)
                                model = KNeighborsClassifier(n_neighbors=n_neighbors) if problem_type == "Classification" else KNeighborsRegressor(n_neighbors=n_neighbors)
                elif model_name == "Decision Tree":
                    if auto_tune:
                        model = DecisionTreeClassifier(random_state=42) if problem_type == "Classification" else DecisionTreeRegressor(random_state=42)
                        param_grid = {"max_depth": [None, 3, 5, 10], "min_samples_split": [2, 5, 10]}
                    else:
                        if manual_params:
                            model = DecisionTreeClassifier(random_state=42, **manual_params) if problem_type == "Classification" else DecisionTreeRegressor(random_state=42, **manual_params)
                        else:
                            if ui_params:
                                dt_params = {k: v for k, v in ui_params.items() if v is not None}
                                model = DecisionTreeClassifier(random_state=42, **dt_params) if problem_type == "Classification" else DecisionTreeRegressor(random_state=42, **dt_params)
                            else:
                                max_depth = st.slider("max_depth", 1, 20, 5)
                                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42) if problem_type == "Classification" else DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                elif model_name == "Naive Bayes":
                    if manual_params:
                        model = GaussianNB(**manual_params)
                    else:
                        model = GaussianNB()
                elif model_name == "Gradient Boosting":
                    if auto_tune:
                        model = GradientBoostingClassifier(random_state=42) if problem_type == "Classification" else GradientBoostingRegressor(random_state=42)
                        param_grid = {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
                    else:
                        if manual_params:
                            model = GradientBoostingClassifier(random_state=42, **manual_params) if problem_type == "Classification" else GradientBoostingRegressor(random_state=42, **manual_params)
                        else:
                            if ui_params:
                                gb_params = {k: v for k, v in ui_params.items() if v is not None}
                                model = GradientBoostingClassifier(random_state=42, **gb_params) if problem_type == "Classification" else GradientBoostingRegressor(random_state=42, **gb_params)
                            else:
                                n_estimators = st.slider("n_estimators", 50, 300, 100)
                                learning_rate = st.number_input("learning_rate", 0.01, 1.0, 0.1)
                                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42) if problem_type == "Classification" else GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                elif model_name == "AdaBoost":
                    if auto_tune:
                        model = AdaBoostClassifier(random_state=42) if problem_type == "Classification" else AdaBoostRegressor(random_state=42)
                        param_grid = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}
                    else:
                        if manual_params:
                            model = AdaBoostClassifier(random_state=42, **manual_params) if problem_type == "Classification" else AdaBoostRegressor(random_state=42, **manual_params)
                        else:
                            if ui_params:
                                n_est = ui_params.get("n_estimators", 50)
                                lr = ui_params.get("learning_rate", 0.1)
                                est_choice = ui_params.get("estimator", "Default")
                                base = None
                                if est_choice == "Decision Tree":
                                    base = DecisionTreeClassifier(random_state=42, max_depth=1) if problem_type == "Classification" else DecisionTreeRegressor(random_state=42, max_depth=1)
                                if base is not None:
                                    model = AdaBoostClassifier(n_estimators=n_est, learning_rate=lr, base_estimator=base, random_state=42) if problem_type == "Classification" else AdaBoostRegressor(n_estimators=n_est, learning_rate=lr, base_estimator=base, random_state=42)
                                else:
                                    model = AdaBoostClassifier(n_estimators=n_est, learning_rate=lr, random_state=42) if problem_type == "Classification" else AdaBoostRegressor(n_estimators=n_est, learning_rate=lr, random_state=42)
                            else:
                                n_estimators = st.slider("n_estimators", 50, 300, 100)
                                learning_rate = st.number_input("learning_rate", 0.01, 1.0, 0.1)
                                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42) if problem_type == "Classification" else AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                elif model_name == "Extra Trees":
                    if auto_tune:
                        model = ExtraTreesClassifier(random_state=42) if problem_type == "Classification" else ExtraTreesRegressor(random_state=42)
                        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
                    else:
                        if manual_params:
                            model = ExtraTreesClassifier(random_state=42, **manual_params) if problem_type == "Classification" else ExtraTreesRegressor(random_state=42, **manual_params)
                        else:
                            n_estimators = st.slider("n_estimators", 50, 300, 100)
                            model = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42) if problem_type == "Classification" else ExtraTreesRegressor(n_estimators=n_estimators, random_state=42)
                else:
                    if manual_params:
                        model = LinearRegression(**manual_params)
                    else:
                        model = LinearRegression()

            # -----------------------------------------------------------
            # 3.2.1 Deep Learning Models: FNN / CNN (optional) / RNN / LSTM
            # -----------------------------------------------------------
            else:
                if problem_type == "Classification":
                    dl_options = ["FNN"]
                    if TF_AVAILABLE:
                        dl_options += ["CNN (optional)", "RNN", "LSTM"]
                    else:
                        st.info(
                            "TensorFlow is not installed in this environment, so CNN, RNN and LSTM "
                            "are unavailable — only FNN (scikit-learn MLP) can be trained. "
                            "Install `tensorflow` to enable the other architectures."
                        )
                else:
                    # Regression: FNN only (no CNN, RNN, LSTM)
                    dl_options = ["FNN (MLP Regressor)"]
                    st.caption("Deep Learning for Regression currently supports FNN architectures.")
                    
                model_name = st.selectbox("Choose deep learning model", dl_options)

                with st.expander("Deep learning architecture settings (optional)"):
                    n_hidden_layers = st.slider("Hidden layers", 1, 4, 2)
                    hidden_units = st.slider("Units per hidden layer", 8, 256, 64)

                is_keras = TF_AVAILABLE and model_name != "FNN" and model_name != "FNN (MLP Regressor)"

                if not is_keras:
                    # FNN via scikit-learn MLPClassifier / MLPRegressor — no external DL dependency required
                    hidden_layer_sizes = tuple([hidden_units] * n_hidden_layers)
                    if problem_type == "Classification":
                        model = MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            learning_rate_init=learning_rate_dl,
                            batch_size=min(int(batch_size), 200),
                            max_iter=int(epoch_limit),
                            tol=max(loss_threshold, 1e-6),
                            n_iter_no_change=max(int(early_stopping_patience), 1),
                            early_stopping=X_val is not None,
                            validation_fraction=0.1,
                            random_state=42,
                        )
                    else:
                        # Regression
                        from sklearn.neural_network import MLPRegressor
                        model = MLPRegressor(
                            hidden_layer_sizes=hidden_layer_sizes,
                            learning_rate_init=learning_rate_dl,
                            batch_size=min(int(batch_size), 200),
                            max_iter=int(epoch_limit),
                            tol=max(loss_threshold, 1e-6),
                            n_iter_no_change=max(int(early_stopping_patience), 1),
                            early_stopping=X_val is not None,
                            validation_fraction=0.1,
                            random_state=42,
                        )

            # ===========================================================
            # Train button
            # ===========================================================
            train_clicked = st.button("Train Model", type="primary")

            if train_clicked and model is not None:
                try:
                    run_label = f"Model {len(st.session_state['training_runs']) + 1}: {model_name}"

                    # ---------------- Classical ML training ----------------
                    if model_family == "Classical ML":
                        if auto_tune and param_grid is not None:
                            scoring = "accuracy" if problem_type == "Classification" else "r2"
                            search = GridSearchCV(
                                model, param_grid, cv=3, scoring=scoring, n_jobs=-1
                            )
                            search.fit(X_train, y_train)
                            model = search.best_estimator_
                            st.write("**Best hyperparameters:**", search.best_params_)
                            st.write(f"**Best CV {scoring}:** {search.best_score_:.3f}")
                        else:
                            model.fit(X_train, y_train)

                        # 3.2.2 Cross-validation settings
                        if use_cv:
                            scoring = "accuracy" if problem_type == "Classification" else "r2"
                            cv_splitter = (
                                StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                                if problem_type == "Classification"
                                else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                            )
                            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring=scoring)
                            st.write(
                                f"**{cv_folds}-fold CV {scoring}:** "
                                f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                            )

                        y_pred = model.predict(X_test)

                    # ---------------- Deep learning training ----------------
                    else:
                        if is_keras:
                            le = LabelEncoder()
                            y_train_enc = le.fit_transform(y_train)
                            y_test_enc = le.transform(y_test)
                            classes_sorted = le.classes_
                            num_classes = len(classes_sorted)

                            if X_val is not None:
                                y_val_enc = le.transform(y_val)

                            if num_classes == 2:
                                y_train_fit = y_train_enc
                                y_val_fit = y_val_enc if X_val is not None else None
                            else:
                                y_train_fit = to_categorical(y_train_enc, num_classes=num_classes)
                                y_val_fit = to_categorical(y_val_enc, num_classes=num_classes) if X_val is not None else None

                            keras_model, loss_fn = build_keras_model(
                                model_name, X_train.shape[1], num_classes, learning_rate_dl
                            )

                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            callbacks = [StProgressCallback(int(epoch_limit), progress_bar, status_text),
                                         TimeoutCallback(timeout_threshold if timeout_threshold > 0 else None)]
                            if loss_threshold > 0:
                                callbacks.append(LossThresholdCallback(loss_threshold))

                            validation_data = (X_val.values, y_val_fit) if X_val is not None else None

                            history = keras_model.fit(
                                X_train.values, y_train_fit,
                                validation_data=validation_data,
                                epochs=int(epoch_limit),
                                batch_size=int(batch_size),
                                callbacks=callbacks,
                                verbose=0,
                            )
                            status_text.text("Training complete.")

                            # 3.2.3 Loss / accuracy curves
                            st.subheader("Training Progress")
                            fig_hist, ax_hist = plt.subplots(1, 2, figsize=(11, 4))
                            ax_hist[0].plot(history.history["loss"], label="Train loss")
                            if "val_loss" in history.history:
                                ax_hist[0].plot(history.history["val_loss"], label="Val loss")
                            ax_hist[0].set_xlabel("Epoch")
                            ax_hist[0].set_ylabel("Loss")
                            ax_hist[0].set_title("Loss Curve")
                            ax_hist[0].legend()

                            ax_hist[1].plot(history.history["accuracy"], label="Train accuracy")
                            if "val_accuracy" in history.history:
                                ax_hist[1].plot(history.history["val_accuracy"], label="Val accuracy")
                            ax_hist[1].set_xlabel("Epoch")
                            ax_hist[1].set_ylabel("Accuracy")
                            ax_hist[1].set_title("Accuracy Curve")
                            ax_hist[1].legend()
                            fig_hist.tight_layout()
                            st.pyplot(fig_hist)

                            model = keras_model  # unify downstream variable name
                            if num_classes == 2:
                                y_score_full = model.predict(X_test.values, verbose=0)
                                y_score = np.hstack([1 - y_score_full, y_score_full])
                                y_pred_enc = (y_score_full.ravel() > 0.5).astype(int)
                            else:
                                y_score = model.predict(X_test.values, verbose=0)
                                y_pred_enc = np.argmax(y_score, axis=1)
                            y_pred = le.inverse_transform(y_pred_enc)

                        else:
                            # FNN via MLPClassifier
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            st.subheader("Training Progress")
                            if hasattr(model, "loss_curve_"):
                                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                                ax_hist.plot(model.loss_curve_, label="Training loss")
                                if hasattr(model, "validation_scores_") and model.validation_scores_:
                                    ax_hist.plot(
                                        np.array(model.validation_scores_) * max(model.loss_curve_),
                                        label="Validation score (scaled)", linestyle="--"
                                    )
                                ax_hist.set_xlabel("Iteration")
                                ax_hist.set_ylabel("Loss")
                                ax_hist.set_title("Loss Curve")
                                ax_hist.legend()
                                st.pyplot(fig_hist)

                    st.success(f"{run_label} trained!")

                    # =======================================================
                    # 3.2.4 Evaluation Metrics
                    # =======================================================
                    if problem_type == "Classification":
                        classes_all = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(y_pred)])))

                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                        cm = confusion_matrix(y_test, y_pred, labels=classes_all)
                        rates = confusion_derived_rates(cm)

                        # Probability scores for AUC / ROC / PR
                        y_score_for_auc = None
                        if is_keras:
                            if len(classes_all) == 2 and y_score.shape[1] == 2:
                                y_score_for_auc = y_score
                            else:
                                y_score_for_auc = y_score
                        elif hasattr(model, "predict_proba"):
                            try:
                                y_score_for_auc = model.predict_proba(X_test)
                            except Exception:
                                y_score_for_auc = None

                        auc_value = None
                        if y_score_for_auc is not None:
                            try:
                                if len(classes_all) == 2:
                                    auc_value = roc_auc_score(y_test, y_score_for_auc[:, 1])
                                else:
                                    auc_value = roc_auc_score(
                                        y_test, y_score_for_auc, multi_class="ovr", average="macro",
                                        labels=classes_all
                                    )
                            except Exception:
                                auc_value = None

                        st.subheader("Evaluation Metrics")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{acc:.3f}")
                        m2.metric("Precision (weighted)", f"{prec:.3f}")
                        m3.metric("Recall (weighted)", f"{rec:.3f}")
                        m4.metric("F1 Score (weighted)", f"{f1:.3f}")

                        m5, m6, m7, m8 = st.columns(4)
                        m5.metric("TPR (macro)", f"{rates['TPR']:.3f}")
                        m6.metric("FPR (macro)", f"{rates['FPR']:.3f}")
                        m7.metric("FNR (macro)", f"{rates['FNR']:.3f}")
                        m8.metric("TNR (macro)", f"{rates['TNR']:.3f}")

                        if auc_value is not None:
                            st.metric("AUC", f"{auc_value:.3f}")
                        else:
                            st.info("AUC unavailable for this model/configuration (no probability scores).")

                        # Confusion matrix
                        st.subheader("Confusion Matrix")
                        fig_cm, ax_cm = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                                    xticklabels=classes_all, yticklabels=classes_all)
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("True")
                        st.pyplot(fig_cm)

                        # ROC curve + PR curve (3.2.4 Visual Outputs)
                        if y_score_for_auc is not None:
                            roc_col, pr_col = st.columns(2)
                            with roc_col:
                                fig_roc, auc_scores = plot_roc_curve(np.array(y_test), y_score_for_auc, classes_all)
                                st.pyplot(fig_roc)
                            with pr_col:
                                fig_pr = plot_pr_curve(np.array(y_test), y_score_for_auc, classes_all)
                                st.pyplot(fig_pr)

                        # Prediction visualization plots
                        st.subheader("Prediction Visualization")
                        fig_pred = plot_prediction_visualization(y_test, y_pred, classes_all)
                        st.pyplot(fig_pred)

                        metrics_record = {
                            "Run": run_label, "Model": model_name, "Family": model_family,
                            "Accuracy": round(acc, 3), "Precision": round(prec, 3),
                            "Recall": round(rec, 3), "F1": round(f1, 3),
                            "TPR": round(rates["TPR"], 3), "FPR": round(rates["FPR"], 3),
                            "FNR": round(rates["FNR"], 3), "TNR": round(rates["TNR"], 3),
                            "AUC": round(auc_value, 3) if auc_value is not None else None,
                        }

                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        st.subheader("Evaluation Metrics")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("MSE", f"{mse:.3f}")
                        m2.metric("MAE", f"{mae:.3f}")
                        m3.metric("R² Score", f"{r2:.3f}")

                        st.subheader("Predicted vs Actual")
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred, alpha=0.7)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
                        ax.set_xlabel("Actual")
                        ax.set_ylabel("Predicted")
                        st.pyplot(fig)

                        metrics_record = {
                            "Run": run_label, "Model": model_name, "Family": model_family,
                            "MSE": round(mse, 3), "MAE": round(mae, 3), "R2": round(r2, 3),
                        }

                    # -------------------------------------------------
                    # 3.2.3 Track model performance across multiple runs
                    # -------------------------------------------------
                    st.session_state["training_runs"].append(metrics_record)
                    st.session_state["trained_models"][run_label] = {
                        "model": model,
                        "is_keras": is_keras,
                        "problem_type": problem_type,
                        "feature_cols": feature_cols,
                        "target_col": target_col,
                        "classes": classes_all if problem_type == "Classification" else None,
                        "label_encoder": le if is_keras else None,
                    }
                    st.session_state["active_run"] = run_label

                except ValueError as e:
                    if problem_type == "Classification":
                        st.error("Please select appropriate target variable.")
                    else:
                        st.error("Training failed because the model could not process the selected data.")
                    st.write("Detailed error:", e)

            # ===========================================================
            # 3.2.3 Multi-model comparison table
            # ===========================================================
            if st.session_state["training_runs"]:
                st.header("4. Model Comparison")
                st.dataframe(pd.DataFrame(st.session_state["training_runs"]), use_container_width=True)
                if st.button("Clear run history"):
                    st.session_state["training_runs"] = []
                    st.session_state["trained_models"] = {}
                    st.session_state["active_run"] = None
                    st.rerun()

            # ===========================================================
            # 3.2.5 Model Import / Export
            # ===========================================================
            st.header("5. Model Import / Export")

            exp_col, imp_col = st.columns(2)

            with exp_col:
                st.subheader("Save a trained model")
                if st.session_state["trained_models"]:
                    export_choice = st.selectbox(
                        "Select a trained run to export",
                        list(st.session_state["trained_models"].keys())
                    )
                    if st.button("Export selected model"):
                        entry = st.session_state["trained_models"][export_choice]
                        os.makedirs("models", exist_ok=True)
                        safe_name = export_choice.replace(" ", "_").replace(":", "")
                        if entry["is_keras"]:
                            file_path = os.path.join("models", f"{safe_name}.keras")
                            entry["model"].save(file_path)
                        else:
                            file_path = os.path.join("models", f"{safe_name}.pkl")
                            joblib.dump(entry["model"], file_path)
                        st.success(f"Saved model to {file_path}")
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()
                        st.download_button(
                            "Download model",
                            data=file_bytes,
                            file_name=os.path.basename(file_path),
                            mime="application/octet-stream"
                        )
                else:
                    st.info("Train a model first to enable export.")

            with imp_col:
                st.subheader("Load a pretrained model")
                pretrained_file = st.file_uploader(
                    "Upload a model file (.pkl / .joblib for classical ML, .keras / .h5 for deep learning)",
                    type=["pkl", "joblib", "keras", "h5"],
                    key="pretrained_uploader"
                )
                if pretrained_file is not None:
                    suffix = pretrained_file.name.split(".")[-1].lower()
                    tmp_path = os.path.join("models", f"uploaded_pretrained.{suffix}")
                    os.makedirs("models", exist_ok=True)
                    with open(tmp_path, "wb") as f:
                        f.write(pretrained_file.getbuffer())
                    try:
                        if suffix in ["keras", "h5"]:
                            if not TF_AVAILABLE:
                                st.error("TensorFlow is required to load Keras models but is not installed.")
                            else:
                                loaded_model = keras_load_model(tmp_path)
                                st.session_state["loaded_pretrained"] = {
                                    "model": loaded_model, "is_keras": True
                                }
                                st.success(f"Loaded Keras model: {pretrained_file.name}")
                        else:
                            loaded_model = joblib.load(tmp_path)
                            st.session_state["loaded_pretrained"] = {
                                "model": loaded_model, "is_keras": False
                            }
                            st.success(f"Loaded model: {pretrained_file.name}")
                    except Exception as e:
                        st.error(f"Could not load model: {e}")

            # ===========================================================
            # 3.2.6 Testing on New Data
            # ===========================================================
            st.header("6. Testing on New Data")

            model_source_options = []
            if st.session_state["trained_models"]:
                model_source_options += [f"Session run: {k}" for k in st.session_state["trained_models"].keys()]
            if st.session_state["loaded_pretrained"] is not None:
                model_source_options.append("Loaded pretrained model")

            if not model_source_options:
                st.info("Train a model or load a pretrained model above to test it on new data.")
            else:
                test_model_choice = st.selectbox("Choose model to use for inference", model_source_options)
                external_file = st.file_uploader(
                    "Upload external test dataset (CSV)", type=["csv"], key="external_test_uploader"
                )

                if external_file is not None:
                    ext_df = pd.read_csv(external_file)
                    st.dataframe(ext_df.head())

                    if test_model_choice.startswith("Session run:"):
                        run_key = test_model_choice.replace("Session run: ", "")
                        entry = st.session_state["trained_models"][run_key]
                        infer_model = entry["model"]
                        infer_is_keras = entry["is_keras"]
                        infer_features = entry["feature_cols"]
                        infer_target = entry["target_col"]
                        infer_le = entry["label_encoder"]
                    else:
                        entry = st.session_state["loaded_pretrained"]
                        infer_model = entry["model"]
                        infer_is_keras = entry["is_keras"]
                        infer_features = [c for c in ext_df.columns if c != target_col]
                        infer_target = target_col if target_col in ext_df.columns else None
                        infer_le = None

                    missing_cols = [c for c in infer_features if c not in ext_df.columns]
                    if missing_cols:
                        st.error(f"Uploaded dataset is missing required feature columns: {missing_cols}")
                    else:
                        X_new = ext_df[infer_features]
                        if st.button("Run inference on uploaded data"):
                            try:
                                if infer_is_keras:
                                    raw_pred = infer_model.predict(X_new.values, verbose=0)
                                    if raw_pred.shape[1] == 1:
                                        pred_enc = (raw_pred.ravel() > 0.5).astype(int)
                                    else:
                                        pred_enc = np.argmax(raw_pred, axis=1)
                                    predictions = infer_le.inverse_transform(pred_enc) if infer_le is not None else pred_enc
                                else:
                                    predictions = infer_model.predict(X_new)

                                result_df = ext_df.copy()
                                result_df["Predicted"] = predictions

                                st.subheader("Predictions")
                                st.dataframe(result_df, use_container_width=True)

                                # 3.2.6 Compare predicted labels with actual labels, if present
                                if infer_target is not None and infer_target in ext_df.columns:
                                    y_true_new = ext_df[infer_target]
                                    acc_new = accuracy_score(y_true_new, predictions)
                                    st.metric("Accuracy on uploaded data", f"{acc_new:.3f}")

                                    cm_new = confusion_matrix(y_true_new, predictions)
                                    fig_new, ax_new = plt.subplots()
                                    sns.heatmap(cm_new, annot=True, fmt="d", cmap="Greens", ax=ax_new)
                                    ax_new.set_xlabel("Predicted")
                                    ax_new.set_ylabel("True")
                                    st.pyplot(fig_new)
                                else:
                                    st.info("No matching target column found in the uploaded data — showing predictions only.")

                                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download predictions (CSV)",
                                    data=csv_bytes,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"Inference failed: {e}")
else:
    st.info("Upload an engineered dataset to start training.")

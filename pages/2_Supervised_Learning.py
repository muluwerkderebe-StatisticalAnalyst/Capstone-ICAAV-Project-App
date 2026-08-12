"""ICAAV automated supervised-machine-learning workflow.

This page capables to do classification/regression model choices,
manual and automatic hyperparameter controls, evaluation plots, and model
download. It adds the complete pipeline stages for exploration, imputation,
categorical encoding, scaling, reusable splitting, model comparison,
cross-validation, model bundles, reload, and live prediction.
"""

from __future__ import annotations

import ast
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


PREFIX = "icaav_sup_"


def state_key(name: str) -> str:
    return f"{PREFIX}{name}"


def parse_hyperparams(param_str: str) -> dict:
    """Parse the original comma-separated manual hyperparameter format."""
    params = {}
    for item in param_str.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            params[key.strip()] = ast.literal_eval(value.strip())
        except Exception:
            params[key.strip()] = value.strip()
    return params


def reset_downstream() -> None:
    """Clear artifacts that are invalid after the dataset/setup changes."""
    for name in [
        "split", "trained_bundle", "comparison", "cv_results", "tuned_bundle",
        "loaded_bundle", "prediction", "preparation",
    ]:
        st.session_state.pop(state_key(name), None)


def infer_problem_type(y: pd.Series) -> str:
    """Suggest classification for categorical or low-cardinality targets."""
    if not pd.api.types.is_numeric_dtype(y):
        return "Classification"
    threshold = max(20, int(len(y) * 0.05))
    return "Classification" if y.nunique(dropna=True) <= threshold else "Regression"


def model_names(problem_type: str) -> list[str]:
    if problem_type == "Classification":
        return [
            "Logistic Regression", "SVM", "Random Forest", "KNN",
            "Decision Tree", "Naive Bayes", "Gradient Boosting",
            "AdaBoost", "Extra Trees",
        ]
    return [
        "Linear Regression", "Random Forest", "KNN", "Decision Tree", "SVR",
        "Gradient Boosting", "AdaBoost", "Extra Trees",
    ]


def make_estimator(problem_type: str, name: str, params: dict | None = None):
    """Create every estimator offered by the original ICAAV page."""
    params = dict(params or {})
    if problem_type == "Classification":
        factories = {
            "Logistic Regression": lambda: LogisticRegression(max_iter=2000, **params),
            "SVM": lambda: SVC(probability=True, **params),
            "Random Forest": lambda: RandomForestClassifier(random_state=42, **params),
            "KNN": lambda: KNeighborsClassifier(**params),
            "Decision Tree": lambda: DecisionTreeClassifier(random_state=42, **params),
            "Naive Bayes": lambda: GaussianNB(**params),
            "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42, **params),
            "AdaBoost": lambda: AdaBoostClassifier(random_state=42, **params),
            "Extra Trees": lambda: ExtraTreesClassifier(random_state=42, **params),
        }
    else:
        factories = {
            "Linear Regression": lambda: LinearRegression(**params),
            "Random Forest": lambda: RandomForestRegressor(random_state=42, **params),
            "KNN": lambda: KNeighborsRegressor(**params),
            "Decision Tree": lambda: DecisionTreeRegressor(random_state=42, **params),
            "SVR": lambda: SVR(**params),
            "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42, **params),
            "AdaBoost": lambda: AdaBoostRegressor(random_state=42, **params),
            "Extra Trees": lambda: ExtraTreesRegressor(random_state=42, **params),
        }
    return factories[name]()


def default_grid(problem_type: str, name: str) -> dict:
    """Small, deployment-friendly grids for the original auto-tune option."""
    common = {
        "Random Forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"],
        },
        "Decision Tree": {
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.1],
        },
        "AdaBoost": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1.0],
        },
        "Extra Trees": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
        },
    }
    if name in common:
        return common[name]
    if name == "Logistic Regression":
        return {"model__C": [0.01, 0.1, 1.0, 10.0]}
    if name in {"SVM", "SVR"}:
        return {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["rbf", "linear"]}
    return {}


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_strategy: str,
    categorical_strategy: str,
    scale_numeric: bool,
):
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=categorical_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(split: dict, name: str, params: dict | None = None) -> Pipeline:
    prep = build_preprocessor(
        split["numeric_cols"], split["categorical_cols"],
        split["numeric_strategy"], split["categorical_strategy"],
        split["scale_numeric"],
    )
    return Pipeline([("preprocessor", prep), ("model", make_estimator(split["problem_type"], name, params))])


def classification_metrics(y_true, y_pred, model=None, X=None) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if model is not None and X is not None and pd.Series(y_true).nunique() == 2:
        try:
            score = model.predict_proba(X)[:, 1]
            metrics["ROC AUC"] = roc_auc_score(y_true, score)
        except Exception:
            pass
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def score_model(problem_type: str, model, X, y) -> tuple[dict, np.ndarray]:
    prediction = model.predict(X)
    if problem_type == "Classification":
        return classification_metrics(y, prediction, model, X), prediction
    return regression_metrics(y, prediction), prediction


def bundle_bytes(bundle: dict) -> bytes:
    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()


def show_metric_cards(metrics: dict) -> None:
    columns = st.columns(min(5, len(metrics)))
    for index, (label, value) in enumerate(metrics.items()):
        columns[index % len(columns)].metric(label, f"{value:.4f}")


def binary_model_scores(model, X) -> np.ndarray | None:
    """Return continuous positive-class scores for a binary ROC curve."""
    try:
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X))
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                return probabilities[:, 1]
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X))
            if scores.ndim == 1:
                return scores
    except Exception:
        return None
    return None


def render_classification_comparison(comparison: dict, split: dict) -> None:
    """Render testing metrics, ROC curves, and a selectable confusion matrix."""
    table = comparison["table"]
    fitted = comparison["models"]
    metric_map = {
        "Test Accuracy": "Accuracy",
        "Test Precision": "Precision",
        "Test Recall": "Recall",
        "Test F1": "F1",
        "Test ROC AUC": "ROC AUC",
    }
    available = [column for column in metric_map if column in table.columns]

    if available:
        chart_df = table[["Model"] + available].dropna(subset=available, how="all")
        long_df = chart_df.melt(
            id_vars="Model", value_vars=available,
            var_name="Metric", value_name="Score",
        )
        long_df["Metric"] = long_df["Metric"].map(metric_map)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        sns.barplot(
            data=long_df, x="Model", y="Score", hue="Metric", ax=ax,
            palette={
                "Accuracy": "#C83349", "Precision": "#4C9AD4",
                "Recall": "#E4A12C", "F1": "#7657E8", "ROC AUC": "#55AD7A",
            },
        )
        ax.set_title("Performance Metrics Across All Models", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="", ncol=min(5, len(available)), loc="upper center")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    y_test = np.asarray(split["y_test"])
    classes = np.unique(y_test)
    left, right = st.columns(2)

    with left:
        st.subheader("ROC Curves — All Models")
        if len(classes) != 2:
            st.info("The combined ROC chart is displayed only for binary classification.")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            plotted = 0
            positive_class = classes[1]
            y_binary = (y_test == positive_class).astype(int)
            for model_name, model in fitted.items():
                scores = binary_model_scores(model, split["X_test"])
                if scores is None:
                    continue
                fpr, tpr, _ = roc_curve(y_binary, scores)
                auc_value = roc_auc_score(y_binary, scores)
                ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} ({auc_value:.3f})")
                plotted += 1
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Random (0.500)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            if plotted:
                ax.legend(fontsize=8, loc="lower right")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("The compared models do not provide probability or decision scores.")
            plt.close(fig)

    with right:
        st.subheader("Confusion Matrix")
        valid_models = list(fitted)
        if valid_models:
            selected = st.selectbox(
                "Model for confusion matrix", valid_models,
                key=state_key("comparison_confusion_model"),
            )
            prediction = fitted[selected].predict(split["X_test"])
            matrix_labels = np.unique(np.concatenate([y_test, np.asarray(prediction)]))
            cm = confusion_matrix(y_test, prediction, labels=matrix_labels)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="RdYlGn",
                xticklabels=matrix_labels, yticklabels=matrix_labels, ax=ax,
            )
            ax.set_title(f"Confusion Matrix — {selected}", fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def render_regression_comparison(comparison: dict, split: dict) -> None:
    """Render regression errors and predicted-versus-actual diagnostics."""
    table = comparison["table"]
    fitted = comparison["models"]
    error_cols = [c for c in ["Test RMSE", "Test MAE"] if c in table.columns]
    if error_cols:
        long_df = table[["Model"] + error_cols].melt(
            id_vars="Model", value_vars=error_cols,
            var_name="Metric", value_name="Error",
        )
        long_df["Metric"] = long_df["Metric"].str.replace("Test ", "", regex=False)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.barplot(data=long_df, x="Model", y="Error", hue="Metric", ax=ax)
        ax.set_title("Regression Error Across All Models", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if fitted:
        selected = st.selectbox(
            "Model for predicted-versus-actual plot", list(fitted),
            key=state_key("comparison_regression_model"),
        )
        actual = np.asarray(split["y_test"])
        predicted = fitted[selected].predict(split["X_test"])
        lower = float(min(np.min(actual), np.min(predicted)))
        upper = float(max(np.max(actual), np.max(predicted)))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(actual, predicted, alpha=0.7, color="#4C9AD4")
        ax.plot([lower, upper], [lower, upper], "--", color="#B31B1B")
        ax.set_title(f"Predicted vs. Actual — {selected}", fontweight="bold")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_evaluation(bundle: dict) -> None:
    split = bundle["split"]
    metrics = bundle["test_metrics"]
    prediction = bundle["test_prediction"]
    show_metric_cards(metrics)
    if split["problem_type"] == "Classification":
        st.subheader("Confusion Matrix")
        labels = np.unique(np.concatenate([np.asarray(split["y_test"]), np.asarray(prediction)]))
        cm = confusion_matrix(split["y_test"], prediction, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
    else:
        st.subheader("Predicted vs. Actual")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(split["y_test"], prediction, alpha=0.7)
        lower = min(np.min(split["y_test"]), np.min(prediction))
        upper = max(np.max(split["y_test"]), np.max(prediction))
        ax.plot([lower, upper], [lower, upper], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
    preview = pd.DataFrame({"Actual": np.asarray(split["y_test"]), "Predicted": prediction})
    st.dataframe(preview.head(25), use_container_width=True)


# ---------------------------------------------------------------------------
# Branding header - preserved ICAAV/Carleton visual identity.
# ---------------------------------------------------------------------------
left, middle, right = st.columns([1, 3, 1])
with left:
    try:
        st.image("assets/icaav_logo.png", width=100)
    except Exception:
        st.markdown("**iCAAV**")
with middle:
    st.markdown(
        """
        <h2 style='text-align:center;color:#B31B1B;margin-bottom:0.2rem;'>
            Automated Supervised Machine Learning Pipeline
        </h2>
        <p style='text-align:center;color:gray;margin-top:0;'>
            Classification • Regression • Evaluation • Prediction
            <br>iCAAV Core • Carleton University
        </p>
        """,
        unsafe_allow_html=True,
    )
with right:
    try:
        st.image("assets/carleton_logo.png", width=100)
    except Exception:
        st.markdown("**Carleton**")

st.markdown("---")


tabs = st.tabs(
    [
        "1. Data Import & Explore",
        "2. Data Preparation",
        "3. Train / Test Split",
        "4. Model Training",
        "5. Model Comparison",
        "6. Validation & Tuning",
        "7. Save, Load & Predict",
    ]
)


# ---------------------------------------------------------------------------
# 1. Data Import & Explore
# ---------------------------------------------------------------------------
with tabs[0]:
    
    uploaded = st.file_uploader("Upload a supervised-learning dataset (CSV)", type=["csv"], key=state_key("upload"))

    if uploaded is not None:
        source_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get(state_key("source_id")) != source_id:
            try:
                df = pd.read_csv(uploaded)
                st.session_state[state_key("df")] = df
                st.session_state[state_key("source_id")] = source_id
                reset_downstream()
            except Exception as exc:
                st.error(f"Could not read the CSV file: {exc}")

    df = st.session_state.get(state_key("df"))
    if df is None:
        st.info("Upload a CSV dataset to begin the automated pipeline.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        with st.expander("Data types and non-null counts"):
            info = pd.DataFrame({
                "Data type": df.dtypes.astype(str),
                "Non-null": df.notna().sum(),
                "Missing": df.isna().sum(),
                "Unique": df.nunique(dropna=True),
            })
            st.dataframe(info, use_container_width=True)
        with st.expander("Descriptive statistics"):
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        target = st.selectbox("Select label/target column", df.columns, key=state_key("target_widget"))
        available_features = [column for column in df.columns if column != target]
        features = st.multiselect(
            "Select feature columns",
            available_features,
            default=available_features,
            key=state_key("features_widget"),
        )
        suggestion = infer_problem_type(df[target])
        problem = st.selectbox(
            "Problem type",
            ["Classification", "Regression"],
            index=0 if suggestion == "Classification" else 1,
            key=state_key("problem_widget"),
            help=f"Suggested from the target: {suggestion}",
        )
        setup = {"target": target, "features": features, "problem_type": problem}
        if st.session_state.get(state_key("setup")) != setup:
            st.session_state[state_key("setup")] = setup
            reset_downstream()

        if not features:
            st.warning("Select at least one feature column.")
        elif problem == "Classification" and df[target].dropna().nunique() < 2:
            st.error("Classification requires at least two target classes.")
        elif problem == "Regression" and not pd.api.types.is_numeric_dtype(df[target]):
            st.error("Regression requires a numeric target column.")
        else:
            st.success(f"Configured {problem.lower()} with {len(features)} feature(s).")


# ---------------------------------------------------------------------------
# 2. Data Preparation
# ---------------------------------------------------------------------------
with tabs[1]:
    
    df = st.session_state.get(state_key("df"))
    setup = st.session_state.get(state_key("setup"))
    if df is None or not setup or not setup["features"]:
        st.info("Complete Step 1 first.")
    else:
        selected = df[setup["features"]]
        numeric_cols = selected.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = [column for column in setup["features"] if column not in numeric_cols]

        c1, c2, c3 = st.columns(3)
        c1.metric("Numeric features", len(numeric_cols))
        c2.metric("Categorical features", len(categorical_cols))
        c3.metric("Rows with missing inputs", int(selected.isna().any(axis=1).sum()))

        st.subheader("Missing-Value Analysis")
        missing = pd.DataFrame({
            "Missing count": selected.isna().sum(),
            "Missing percent": (selected.isna().mean() * 100).round(2),
        })
        st.dataframe(missing, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        numeric_strategy = col1.selectbox("Numeric imputation", ["median", "mean", "most_frequent"], key=state_key("num_impute"))
        categorical_strategy = col2.selectbox("Categorical imputation", ["most_frequent", "constant"], key=state_key("cat_impute"))
        scale_numeric = col3.checkbox("Standardize numeric features", value=True, key=state_key("scale"))
        drop_duplicates = st.checkbox("Remove duplicate rows before splitting", value=True, key=state_key("drop_duplicates"))

        preparation = {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "numeric_strategy": numeric_strategy,
            "categorical_strategy": categorical_strategy,
            "scale_numeric": scale_numeric,
            "drop_duplicates": drop_duplicates,
        }
        if st.session_state.get(state_key("preparation")) != preparation:
            st.session_state[state_key("preparation")] = preparation
            for name in ["split", "trained_bundle", "comparison", "cv_results", "tuned_bundle", "prediction"]:
                st.session_state.pop(state_key(name), None)

        st.subheader("Automatic Pipeline Preview")
        st.write(f"Numeric: impute with **{numeric_strategy}**" + (" and standardize." if scale_numeric else "."))
        st.write(f"Categorical: impute with **{categorical_strategy}** and one-hot encode unknown-safe categories.")
        st.caption("The preprocessor is fitted only on training data to reduce leakage.")


# ---------------------------------------------------------------------------
# 3. Train/Test Split
# ---------------------------------------------------------------------------
with tabs[2]:
   
    df = st.session_state.get(state_key("df"))
    setup = st.session_state.get(state_key("setup"))
    preparation = st.session_state.get(state_key("preparation"))
    if df is None or not setup or not preparation:
        st.info("Complete Steps 1 and 2 first.")
    else:
        c1, c2 = st.columns(2)
        test_size = c1.slider("Test size (%)", 10, 40, 20, key=state_key("test_size")) / 100
        random_state = c2.number_input("Random state", min_value=0, value=42, step=1, key=state_key("random_state"))
        stratify = st.checkbox(
            "Stratify classification targets when possible",
            value=True,
            disabled=setup["problem_type"] != "Classification",
            key=state_key("stratify"),
        )

        if st.button("Split the Dataset", type="primary", key=state_key("split_button")):
            try:
                working = df[setup["features"] + [setup["target"]]].copy()
                working = working.dropna(subset=[setup["target"]])
                if preparation["drop_duplicates"]:
                    working = working.drop_duplicates()
                X = working[setup["features"]]
                y = working[setup["target"]]
                stratify_values = None
                if setup["problem_type"] == "Classification" and stratify and y.value_counts().min() >= 2:
                    stratify_values = y
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state), stratify=stratify_values
                )
                split = {
                    **setup, **preparation,
                    "X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test,
                    "test_size": test_size, "random_state": int(random_state),
                }
                st.session_state[state_key("split")] = split
                for name in ["trained_bundle", "comparison", "cv_results", "tuned_bundle", "prediction"]:
                    st.session_state.pop(state_key(name), None)
                st.success("Dataset split successfully.")
            except Exception as exc:
                st.error(f"Could not split the dataset: {exc}")

        split = st.session_state.get(state_key("split"))
        if split:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Training rows", f"{len(split['X_train']):,}")
            c2.metric("Testing rows", f"{len(split['X_test']):,}")
            c3.metric("Features", len(split["features"]))
            c4.metric("Target classes", split["y_train"].nunique() if split["problem_type"] == "Classification" else "Continuous")
            if split["problem_type"] == "Classification":
                counts = pd.concat(
                    [split["y_train"].value_counts(normalize=True).rename("Train share"),
                     split["y_test"].value_counts(normalize=True).rename("Test share")], axis=1
                ).fillna(0)
                st.dataframe(counts, use_container_width=True)


# ---------------------------------------------------------------------------
# 4. Model Training
# ---------------------------------------------------------------------------
with tabs[3]:
    
    split = st.session_state.get(state_key("split"))
    if split is None:
        st.info("Create the train/test split in Step 3 first.")
    else:
        name = st.selectbox("Choose model", model_names(split["problem_type"]), key=state_key("model_name"))
        training_mode = st.selectbox("Training mode", ["Manual", "Auto-tune hyperparameters"], key=state_key("training_mode"))

        ui_params = {}
        if training_mode == "Manual":
            with st.expander("Set hyperparameters for selected model (optional)", expanded=True):
                if name in {"Random Forest", "Extra Trees"}:
                    ui_params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, key=state_key("trees"))
                    depth = st.selectbox("max_depth", ["None", 3, 5, 10, 20], key=state_key("depth"))
                    ui_params["max_depth"] = None if depth == "None" else int(depth)
                    ui_params["min_samples_split"] = int(st.number_input("min_samples_split", 2, 100, 2, key=state_key("min_split")))
                elif name == "KNN":
                    ui_params["n_neighbors"] = st.slider("n_neighbors", 1, 50, 5, key=state_key("neighbors"))
                    ui_params["weights"] = st.selectbox("weights", ["uniform", "distance"], key=state_key("weights"))
                    ui_params["metric"] = st.selectbox("metric", ["minkowski", "euclidean", "manhattan"], key=state_key("metric"))
                elif name == "Decision Tree":
                    if split["problem_type"] == "Classification":
                        ui_params["criterion"] = st.selectbox("criterion", ["gini", "entropy", "log_loss"], key=state_key("criterion"))
                    depth = st.selectbox("max_depth", ["None", 3, 5, 10, 20], key=state_key("tree_depth"))
                    ui_params["max_depth"] = None if depth == "None" else int(depth)
                    ui_params["min_samples_leaf"] = int(st.number_input("min_samples_leaf", 1, 100, 1, key=state_key("min_leaf")))
                elif name in {"SVM", "SVR"}:
                    ui_params["C"] = st.number_input("C", 0.01, 100.0, 1.0, key=state_key("C"))
                    ui_params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"], key=state_key("kernel"))
                    if name == "SVR":
                        ui_params["epsilon"] = st.number_input("epsilon", 0.0, 1.0, 0.1, key=state_key("epsilon"))
                elif name in {"Gradient Boosting", "AdaBoost"}:
                    ui_params["learning_rate"] = st.number_input("learning_rate", 0.001, 1.0, 0.1, key=state_key("learning_rate"))
                    ui_params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, key=state_key("boost_estimators"))

            manual_text = st.text_input(
                "Advanced manual hyperparameters",
                help="Comma-separated values such as C=1.0, max_depth=5. These override the widgets.",
                key=state_key("manual_text"),
            )
            ui_params.update(parse_hyperparams(manual_text))

        if st.button("Train Model", type="primary", key=state_key("train_button")):
            try:
                pipeline = build_model_pipeline(split, name, ui_params if training_mode == "Manual" else None)
                best_params = None
                if training_mode == "Auto-tune hyperparameters" and default_grid(split["problem_type"], name):
                    scoring = "accuracy" if split["problem_type"] == "Classification" else "r2"
                    search = GridSearchCV(pipeline, default_grid(split["problem_type"], name), cv=3, scoring=scoring, n_jobs=-1)
                    search.fit(split["X_train"], split["y_train"])
                    pipeline = search.best_estimator_
                    best_params = search.best_params_
                else:
                    pipeline.fit(split["X_train"], split["y_train"])
                train_metrics, train_prediction = score_model(split["problem_type"], pipeline, split["X_train"], split["y_train"])
                test_metrics, test_prediction = score_model(split["problem_type"], pipeline, split["X_test"], split["y_test"])
                bundle = {
                    "model": pipeline, "model_name": name, "problem_type": split["problem_type"],
                    "target": split["target"], "feature_cols": split["features"],
                    "train_metrics": train_metrics, "test_metrics": test_metrics,
                    "train_prediction": train_prediction, "test_prediction": test_prediction,
                    "best_params": best_params, "split": split,
                    "feature_examples": st.session_state[state_key("df")][split["features"]].copy(),
                }
                st.session_state[state_key("trained_bundle")] = bundle
                # Compatibility with the original Page 3 transfer workflow.
                st.session_state["trained"] = {
                    "model": pipeline, "model_name": name, "problem_type": split["problem_type"],
                    "feature_cols": split["features"], "X_test": split["X_test"], "y_test": split["y_test"],
                }
                st.success("Model trained successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        trained = st.session_state.get(state_key("trained_bundle"))
        if trained:
            if trained.get("best_params"):
                st.write("**Best hyperparameters:**", trained["best_params"])
            st.subheader("Training Metrics")
            show_metric_cards(trained["train_metrics"])
            st.subheader("Testing-Set Performance")
            render_evaluation(trained)


# ---------------------------------------------------------------------------
# 5. Model Comparison
# ---------------------------------------------------------------------------
with tabs[4]:

    split = st.session_state.get(state_key("split"))
    if split is None:
        st.info("Create the train/test split in Step 3 first.")
    else:
        defaults = ["Logistic Regression", "Decision Tree", "Random Forest"] if split["problem_type"] == "Classification" else ["Linear Regression", "Decision Tree", "Random Forest"]
        selected_models = st.multiselect(
            "Models to compare",
            model_names(split["problem_type"]),
            default=defaults,
            key=state_key("comparison_models"),
        )
        if st.button("Train and Compare Models", type="primary", key=state_key("compare_button")):
            results, fitted = [], {}
            progress = st.progress(0.0)
            for index, model_name in enumerate(selected_models):
                try:
                    pipeline = build_model_pipeline(split, model_name)
                    pipeline.fit(split["X_train"], split["y_train"])
                    train_metrics, _ = score_model(split["problem_type"], pipeline, split["X_train"], split["y_train"])
                    test_metrics, _ = score_model(split["problem_type"], pipeline, split["X_test"], split["y_test"])
                    row = {"Model": model_name}
                    row.update({f"Train {key}": value for key, value in train_metrics.items()})
                    row.update({f"Test {key}": value for key, value in test_metrics.items()})
                    results.append(row)
                    fitted[model_name] = pipeline
                except Exception as exc:
                    results.append({"Model": model_name, "Error": str(exc)})
                progress.progress((index + 1) / max(1, len(selected_models)))
            st.session_state[state_key("comparison")] = {"table": pd.DataFrame(results), "models": fitted}

        comparison = st.session_state.get(state_key("comparison"))
        if comparison:
            table = comparison["table"]
            valid_table = table[table.get("Error", pd.Series(index=table.index, dtype=object)).isna()].copy()

            if split["problem_type"] == "Classification" and "Test Accuracy" in valid_table.columns:
                best_row = valid_table.loc[valid_table["Test Accuracy"].idxmax()]
                st.subheader(f"Best Testing Model: {best_row['Model']}")
                card_metrics = [
                    ("Accuracy", "Test Accuracy"),
                    ("Precision", "Test Precision"),
                    ("Recall", "Test Recall"),
                    ("F1 Score", "Test F1"),
                    ("ROC AUC", "Test ROC AUC"),
                ]
                cards = st.columns(5)
                for card, (label, column) in zip(cards, card_metrics):
                    value = best_row.get(column, np.nan)
                    display = "N/A" if pd.isna(value) else (
                        f"{value:.3f}" if label == "ROC AUC" else f"{value:.1%}"
                    )
                    card.metric(label, display)
            elif split["problem_type"] == "Regression" and "Test MAE" in valid_table.columns:
                best_row = valid_table.loc[valid_table["Test MAE"].idxmin()]
                st.subheader(f"Lowest-MAE Testing Model: {best_row['Model']}")
                cards = st.columns(4)
                for card, column in zip(cards, ["Test MAE", "Test RMSE", "Test MSE", "Test R2"]):
                    value = best_row.get(column, np.nan)
                    card.metric(column.replace("Test ", ""), "N/A" if pd.isna(value) else f"{value:.4f}")

            st.subheader("All Models vs. Testing Metrics")
            test_columns = ["Model"] + [c for c in table.columns if c.startswith("Test ")] + (["Error"] if "Error" in table else [])
            st.dataframe(table[test_columns], use_container_width=True)

            if split["problem_type"] == "Classification":
                render_classification_comparison(comparison, split)
            else:
                render_regression_comparison(comparison, split)

            st.download_button(
                "Download Model Comparison CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name="icaav_model_comparison.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# 6. Validation & Tuning
# ---------------------------------------------------------------------------
with tabs[5]:
   
    split = st.session_state.get(state_key("split"))
    if split is None:
        st.info("Create the train/test split in Step 3 first.")
    else:
        validation_name = st.selectbox("Model to validate", model_names(split["problem_type"]), key=state_key("validation_model"))
        folds = st.slider("Cross-validation folds", 2, 10, 5, key=state_key("folds"))
        if st.button("Run Cross-Validation", key=state_key("cv_button")):
            try:
                pipeline = build_model_pipeline(split, validation_name)
                scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"] if split["problem_type"] == "Classification" else ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
                results = cross_validate(pipeline, split["X_train"], split["y_train"], cv=folds, scoring=scoring, n_jobs=-1)
                summary = []
                for key, values in results.items():
                    if not key.startswith("test_"):
                        continue
                    display_values = -values if key in {"test_neg_mean_absolute_error", "test_neg_root_mean_squared_error"} else values
                    summary.append({"Metric": key.replace("test_", "").replace("neg_", ""), "Mean": np.mean(display_values), "Std": np.std(display_values)})
                st.session_state[state_key("cv_results")] = pd.DataFrame(summary)
            except Exception as exc:
                st.error(f"Cross-validation failed: {exc}")

        cv_results = st.session_state.get(state_key("cv_results"))
        if cv_results is not None:
            st.dataframe(cv_results, use_container_width=True)

        st.subheader("Grid-Search Tuning")
        grid = default_grid(split["problem_type"], validation_name)
        if not grid:
            st.info("This model has no predefined grid. Use its manual controls in Step 4.")
        else:
            st.json(grid)
            if st.button("Tune Selected Model", type="primary", key=state_key("tune_button")):
                try:
                    pipeline = build_model_pipeline(split, validation_name)
                    scoring = "accuracy" if split["problem_type"] == "Classification" else "r2"
                    search = GridSearchCV(pipeline, grid, cv=folds, scoring=scoring, n_jobs=-1)
                    search.fit(split["X_train"], split["y_train"])
                    test_metrics, test_prediction = score_model(split["problem_type"], search.best_estimator_, split["X_test"], split["y_test"])
                    bundle = {
                        "model": search.best_estimator_, "model_name": validation_name,
                        "problem_type": split["problem_type"], "target": split["target"],
                        "feature_cols": split["features"], "test_metrics": test_metrics,
                        "test_prediction": test_prediction, "split": split,
                        "best_params": search.best_params_,
                        "feature_examples": st.session_state[state_key("df")][split["features"]].copy(),
                    }
                    st.session_state[state_key("tuned_bundle")] = bundle
                except Exception as exc:
                    st.error(f"Grid search failed: {exc}")

        tuned = st.session_state.get(state_key("tuned_bundle"))
        if tuned:
            st.write("**Best parameters:**", tuned["best_params"])
            render_evaluation(tuned)


# ---------------------------------------------------------------------------
# 7. Save, Load & Predict
# ---------------------------------------------------------------------------
with tabs[6]:
    
    trained = st.session_state.get(state_key("trained_bundle"))
    tuned = st.session_state.get(state_key("tuned_bundle"))
    comparison = st.session_state.get(state_key("comparison"))

    available = {}
    if trained:
        available[f"Trained - {trained['model_name']}"] = trained
    if tuned:
        available[f"Tuned - {tuned['model_name']}"] = tuned
    if comparison:
        split = st.session_state.get(state_key("split"))
        source_df = st.session_state.get(state_key("df"))
        for name, model in comparison["models"].items():
            available[f"Compared - {name}"] = {
                "model": model, "model_name": name, "problem_type": split["problem_type"],
                "target": split["target"], "feature_cols": split["features"],
                "feature_examples": source_df[split["features"]].copy(),
            }

    st.subheader("Save a Complete Model Bundle")
    if available:
        save_choice = st.selectbox("Model to save", list(available), key=state_key("save_choice"))
        st.download_button(
            "Download ICAAV Supervised Model (.pkl)",
            data=bundle_bytes(available[save_choice]),
            file_name="icaav_supervised_model_bundle.pkl",
            mime="application/octet-stream",
        )
    else:
        st.info("Train, tune, or compare a model before saving.")

    st.subheader("Load a Saved Model Bundle")
    st.warning("Only load pickle/Joblib files from a trusted source.")
    model_file = st.file_uploader("Upload a saved ICAAV model bundle", type=["pkl", "joblib"], key=state_key("model_upload"))
    if model_file is not None and st.button("Load Model Bundle", key=state_key("load_button")):
        try:
            loaded = joblib.load(BytesIO(model_file.read()))
            required = {"model", "model_name", "problem_type", "feature_cols", "feature_examples"}
            if not isinstance(loaded, dict) or not required.issubset(loaded):
                raise ValueError("The file is not a compatible ICAAV model bundle.")
            st.session_state[state_key("loaded_bundle")] = loaded
            st.success(f"Loaded {loaded['model_name']} successfully.")
        except Exception as exc:
            st.error(f"Could not load the model bundle: {exc}")

    loaded = st.session_state.get(state_key("loaded_bundle"))
    if loaded:
        available[f"Uploaded - {loaded['model_name']}"] = loaded

    st.subheader("Live Prediction")
    if not available:
        st.info("A trained or loaded model is required for prediction.")
    else:
        predict_choice = st.radio("Predict using", list(available), horizontal=True, key=state_key("predict_choice"))
        active = available[predict_choice]
        examples = active["feature_examples"]
        values = {}
        with st.form("icaav_supervised_prediction_form"):
            for feature in active["feature_cols"]:
                series = examples[feature]
                if pd.api.types.is_numeric_dtype(series):
                    default = float(series.dropna().median()) if not series.dropna().empty else 0.0
                    values[feature] = st.number_input(feature, value=default)
                else:
                    choices = series.dropna().astype(str).unique().tolist()
                    if not choices:
                        choices = [""]
                    values[feature] = st.selectbox(feature, choices)
            submitted = st.form_submit_button("Make Prediction", type="primary")

        if submitted:
            try:
                row = pd.DataFrame([values], columns=active["feature_cols"])
                prediction = active["model"].predict(row)[0]
                result = {"Prediction": prediction}
                if active["problem_type"] == "Classification" and hasattr(active["model"], "predict_proba"):
                    probabilities = active["model"].predict_proba(row)[0]
                    classes = active["model"].classes_
                    result["Probabilities"] = {str(label): float(probability) for label, probability in zip(classes, probabilities)}
                st.session_state[state_key("prediction")] = result
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        result = st.session_state.get(state_key("prediction"))
        if result:
            st.success(f"Model prediction: **{result['Prediction']}**")
            if "Probabilities" in result:
                st.dataframe(pd.DataFrame.from_dict(result["Probabilities"], orient="index", columns=["Probability"]), use_container_width=True)

    st.caption("Educational analytical output only. Validate the model and data before operational use.")

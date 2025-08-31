# apps/app.py
import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import Bunch

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = joblib.load(model_path)
    return model

def get_preprocessor_and_feature_lists(pipeline: Pipeline):
    """
    Returns (preprocessor, numeric_cols, categorical_cols, other_cols)
    Works when pipeline contains a step named 'preprocessor' (ColumnTransformer).
    """
    if "preprocessor" in pipeline.named_steps:
        preproc = pipeline.named_steps["preprocessor"]
    else:
        # try to find ColumnTransformer in steps
        preproc = None
        for _, step in pipeline.steps:
            if isinstance(step, ColumnTransformer):
                preproc = step
                break
        if preproc is None:
            raise ValueError("Could not find ColumnTransformer named 'preprocessor' in pipeline.")

    # Extract columns specified in the ColumnTransformer
    numeric_cols = []
    categorical_cols = []
    other_cols = []
    try:
        for name, transformer, cols in preproc.transformers:
            # Some transformers may be ('remainder', 'drop'/'passthrough', <...>)
            if name == "num" or (isinstance(transformer, Pipeline) and "imputer" in str(transformer).lower()):
                # assume numeric pipeline
                numeric_cols.extend(list(cols))
            elif name == "cat":
                categorical_cols.extend(list(cols))
            else:
                # remainder or other
                other_cols.extend(list(cols) if isinstance(cols, (list, tuple, np.ndarray)) else [])
    except Exception:
        # fallback: try named_transformers_
        try:
            numeric_cols = list(preproc.transformers_[0][2])
            categorical_cols = list(preproc.transformers_[1][2])
        except Exception:
            pass

    # remove None and ensure unique
    numeric_cols = [c for c in (numeric_cols or []) if c is not None]
    categorical_cols = [c for c in (categorical_cols or []) if c is not None]
    return preproc, numeric_cols, categorical_cols, other_cols

def safe_get_imputer_stats(preproc, which="num"):
    """
    Returns array of imputer statistics for numeric or categorical pipeline if available.
    """
    try:
        named = preproc.named_transformers_
        if which == "num" and "num" in named:
            imputer = named["num"].named_steps.get("imputer", None)
            if imputer is not None and hasattr(imputer, "statistics_"):
                return np.array(imputer.statistics_)
        if which == "cat" and "cat" in named:
            imputer = named["cat"].named_steps.get("imputer", None)
            if imputer is not None and hasattr(imputer, "statistics_"):
                return np.array(imputer.statistics_, dtype=object)
    except Exception:
        pass
    return None

def safe_get_ohe_categories(preproc):
    """
    Returns list of arrays of categories for each categorical column if OneHotEncoder was used.
    """
    try:
        named = preproc.named_transformers_
        if "cat" in named:
            onehot = named["cat"].named_steps.get("onehot", None)
            if onehot is not None and hasattr(onehot, "categories_"):
                return list(onehot.categories_)
    except Exception:
        pass
    return None

def get_feature_names(preproc):
    """
    Get output feature names after preprocessing (useful for feature importance mapping).
    """
    try:
        # sklearn >= 1.0 supports get_feature_names_out
        return preproc.get_feature_names_out()
    except Exception:
        # fallback: attempt to build names manually
        names = []
        try:
            # numeric columns
            num_cols = preproc.transformers[0][2]
            if num_cols:
                names.extend(list(num_cols))
            # categorical: expand with categories if available
            cat_cols = preproc.transformers[1][2]
            onehot = preproc.named_transformers_["cat"].named_steps.get("onehot", None)
            if onehot is not None and hasattr(onehot, "categories_"):
                cats = onehot.categories_
                for col, catlist in zip(cat_cols, cats):
                    names.extend([f"{col}__{str(x)}" for x in catlist])
            else:
                # fallback: keep original cat col names
                names.extend(list(cat_cols))
        except Exception:
            pass
        return np.array(names)

def pretty_prediction(pred, prob):
    label = "Default" if pred == 1 else "No Default"
    prob_pct = round(prob * 100, 2)
    return label, prob_pct

# -------------------------
# Load model
# -------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
MODEL_PATH = ROOT / "models" / "loan_default_model.pkl"

st.title("💳 Loan Default Predictor")
st.markdown(
    """
Live demo of the Loan Default Predictor model — enter loan/customer details in the sidebar and click **Predict**.
This app loads your trained pipeline (preprocessor + model) dynamically and builds the input form automatically.
"""
)

try:
    pipeline = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# extract preprocessor & feature lists
try:
    preproc, numeric_cols, categorical_cols, other_cols = get_preprocessor_and_feature_lists(pipeline)
except Exception as e:
    st.error(f"Error extracting preprocessor/columns from pipeline: {e}")
    st.stop()

# get imputer stats & categories for sensible defaults
num_stats = safe_get_imputer_stats(preproc, "num")
cat_stats = safe_get_imputer_stats(preproc, "cat")
ohe_categories = safe_get_ohe_categories(preproc)
feature_names_after_preproc = None
try:
    feature_names_after_preproc = get_feature_names(preproc)
except Exception:
    feature_names_after_preproc = None

# Sidebar UI for inputs
st.sidebar.header("Input loan features")

use_sample = st.sidebar.checkbox("Auto-fill sample (median/mode)", value=True)

input_data = {}
with st.sidebar.form(key="input_form"):
    st.write("### Numeric features")
    for i, col in enumerate(numeric_cols):
        default = float(num_stats[i]) if (num_stats is not None and i < len(num_stats) and not pd.isna(num_stats[i])) else 0.0
        # reasonable ranges if default is zero
        min_val = -1e6
        max_val = 1e9
        step = 1.0 if default >= 1 else 0.01
        input_data[col] = st.number_input(label=col, value=default if use_sample else 0.0, format="%.2f")
    st.write("### Categorical features")
    # prepare categories mapping
    if categorical_cols:
        # if we have onehot categories, map them
        if ohe_categories is not None and len(ohe_categories) == len(categorical_cols):
            for col, cats in zip(categorical_cols, ohe_categories):
                opts = [str(x) for x in cats]
                default = cat_stats[[i for i, c in enumerate(categorical_cols) if c == col][0]] if (cat_stats is not None) else opts[0] if len(opts) else ""
                default = str(default) if use_sample else opts[0] if len(opts) else ""
                input_data[col] = st.selectbox(label=col, options=opts, index=opts.index(default) if default in opts else 0)
        else:
            # unknown categories, allow free text
            for col in categorical_cols:
                default = str(cat_stats[[i for i, c in enumerate(categorical_cols) if c == col][0]]) if (cat_stats is not None) else ""
                input_data[col] = st.text_input(label=col, value=default if use_sample else "")
    st.write("---")
    st.form_submit_button("Predict")

# allow CSV upload for batch predictions
st.sidebar.write("### Batch prediction")
uploaded = st.sidebar.file_uploader("Upload CSV with same raw columns", type=["csv"])
st.sidebar.write("CSV should contain raw columns (not one-hot encoded). We will run preprocessing before prediction.")

# -------------------------
# Main: Single record prediction
# -------------------------
st.subheader("Single prediction")

if st.button("Run Prediction on current inputs"):
    try:
        # Build single-row DataFrame in same column order expected by pipeline
        X_input = pd.DataFrame([input_data], columns=(numeric_cols + categorical_cols))
        # Ensure dtypes: numeric to numeric
        for nc in numeric_cols:
            if nc in X_input.columns:
                X_input[nc] = pd.to_numeric(X_input[nc], errors="coerce")
        # Predict
        pred = pipeline.predict(X_input)[0]
        prob = pipeline.predict_proba(X_input)[0][1] if hasattr(pipeline, "predict_proba") else None
        label, prob_pct = pretty_prediction(pred, prob if prob is not None else 0.0)

        # Display results
        col1, col2 = st.columns([2, 3])
        with col1:
            st.metric(label="Prediction", value=label, delta=f"{prob_pct}% prob")
            st.write("**Raw model output:**")
            st.write(dict(prediction=int(pred), probability=float(prob) if prob is not None else None))
        with col2:
            st.info("Model info:")
            st.write(f"- Pipeline steps: `{[name for name, _ in pipeline.steps]}`")
            model_name = type(pipeline.named_steps.get("model", pipeline.steps[-1][1])).__name__
            st.write(f"- Model type: **{model_name}**")
            if hasattr(pipeline.named_steps.get("model", None), "classes_"):
                st.write(f"- Classes: {pipeline.named_steps.get('model').classes_}")

        # Feature contributions (if logistic or tree)
        try:
            model_step = pipeline.named_steps.get("model", pipeline.steps[-1][1])
            # get processed feature names (after preprocessing)
            if feature_names_after_preproc is None:
                feature_names_after_preproc = get_feature_names(preproc)
            # get transformed single-row vector
            X_trans = pipeline.named_steps["preprocessor"].transform(X_input)
            # If sparse -> toarray
            try:
                X_trans_arr = X_trans.toarray() if hasattr(X_trans, "toarray") else np.array(X_trans)
            except Exception:
                X_trans_arr = np.array(X_trans)
            # Logistic regression: coef * x
            if model_name.lower().startswith("logistic"):
                coefs = model_step.coef_.flatten()
                contrib = coefs * X_trans_arr.flatten()
                feat_names = feature_names_after_preproc if feature_names_after_preproc is not None else np.arange(len(contrib))
                df_contrib = pd.DataFrame({"feature": feat_names, "contribution": contrib})
                df_top_pos = df_contrib.sort_values("contribution", ascending=False).head(8)
                df_top_neg = df_contrib.sort_values("contribution", ascending=True).head(8)

                st.markdown("**Top positive contributions → increase default probability**")
                st.table(df_top_pos.set_index("feature").round(4))
                st.markdown("**Top negative contributions → decrease default probability**")
                st.table(df_top_neg.set_index("feature").round(4))

            # Decision Tree: feature_importances_
            elif hasattr(model_step, "feature_importances_"):
                importances = model_step.feature_importances_
                feat_names = feature_names_after_preproc if feature_names_after_preproc is not None else np.arange(len(importances))
                df_imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
                st.markdown("**Top feature importances (Decision Tree)**")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=df_imp, x="importance", y="feature", ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as ex:
            st.warning(f"Could not compute feature contributions: {ex}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------
# Main: Batch predictions from CSV
# -------------------------
st.subheader("Batch prediction (CSV)")

if uploaded is not None:
    try:
        df_uploaded = pd.read_csv(uploaded)
        st.write(f"Uploaded {df_uploaded.shape[0]} rows and {df_uploaded.shape[1]} columns.")
        # Basic validation: check required columns exist
        expected_cols = numeric_cols + categorical_cols
        missing = [c for c in expected_cols if c not in df_uploaded.columns]
        if missing:
            st.error(f"Uploaded CSV missing required columns: {missing}")
        else:
            with st.spinner("Running preprocessing and predictions ..."):
                preds = pipeline.predict(df_uploaded[expected_cols])
                probs = pipeline.predict_proba(df_uploaded[expected_cols])[:, 1] if hasattr(pipeline, "predict_proba") else np.zeros(len(preds))
                df_uploaded["pred_default"] = preds
                df_uploaded["prob_default"] = probs
            st.success("Predictions added to uploaded DataFrame")
            st.dataframe(df_uploaded.head(100))
            # allow download
            csv = df_uploaded.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

# -------------------------
# Diagnostics & Info
# -------------------------
st.sidebar.write("---")
st.sidebar.header("Model diagnostics")
st.sidebar.write(f"Preprocessor transforms: numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
try:
    model_step = pipeline.named_steps.get("model", pipeline.steps[-1][1])
    st.sidebar.write(f"Model: `{type(model_step).__name__}`")
    # show if classifier has feature_importances_
    if hasattr(model_step, "feature_importances_"):
        st.sidebar.write("- Decision-tree style model with feature_importances_")
    elif hasattr(model_step, "coef_"):
        st.sidebar.write("- Linear model with coefficients")
except Exception:
    pass

st.markdown("---")
st.markdown("### Notes & Tips")
st.markdown(
    """
- This app loads the **trained pipeline** (preprocessing + model) you saved after training.
- If the app errors on missing columns, confirm your uploaded CSV contains the raw columns the pipeline expects (not one-hot encoded columns).
- For reproducible results, always use the same preprocessing and model versions.
"""
)

# Footer
st.markdown("---")
st.write("Built with ❤️ — streamlit. For issues, check the project notebooks in `notebooks/`.")

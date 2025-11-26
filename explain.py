import shap
import joblib
import pandas as pd
from .config import RISK_MODEL_PATH


def explain_xgb_model(X: pd.DataFrame, model_path: str = None):
    model = joblib.load(model_path or RISK_MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer

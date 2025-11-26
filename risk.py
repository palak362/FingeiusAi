import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from ..config import RISK_MODEL_PATH


def train_risk_model(df: pd.DataFrame, target_col="default_flag"):
    # Drop non-numeric columns
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # Ensure everything is numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    joblib.dump(model, RISK_MODEL_PATH)

    return {
        "model": model,
        "auc": auc,
        "report": classification_report(y_test, model.predict(X_test), output_dict=True)
    }

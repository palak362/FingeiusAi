import streamlit as st
import pandas as pd
import plotly.express as px
from src.pdf_extract import extract_transactions_from_pdf
from src.data_loader import load_processed
from src.models.risk import train_risk_model
from src.explain import explain_xgb_model
from pathlib import Path
from src.config import RISK_MODEL_PATH


st.set_page_config(page_title="FinGenius AI", layout="wide")


st.title("FinGenius AI — Unified Risk, Fraud & Financial Intelligence Engine")


tabs = st.tabs(["Dashboard", "PDF Upload & Extract", "Modeling", "Explainability", "Download"])


# ---------------- Dashboard ----------------
with tabs[0]:
    st.header("Overview Dashboard")
    try:
        df = load_processed()
    except Exception:
        st.info("No processed data found. Upload a PDF or CSV in 'PDF Upload & Extract' tab.")
        df = None

    if df is not None and not df.empty:
        st.subheader("Transactions sample")
        st.write(df.head())

        if "amount" in df.columns:
            fig = px.histogram(df, x="amount", nbins=60, title="Transaction Amount Distribution")
            st.plotly_chart(fig, use_container_width=True)

        if "description" in df.columns:
            top = df.groupby("description", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(10)
            st.bar_chart(top.set_index("description")["amount"])


# --------------- PDF Upload & Extract ----------------
with tabs[1]:
    st.header("Upload bank statement PDF or CSV")
    pdf = st.file_uploader(
        "Upload PDF (bank statement) or CSV of transactions", type=["pdf", "csv"]
    )

    if pdf is not None:
        if pdf.type == "application/pdf":
            with open("tmp_statement.pdf", "wb") as f:
                f.write(pdf.getbuffer())
            df_txn = extract_transactions_from_pdf("tmp_statement.pdf")
        else:
            df_txn = pd.read_csv(pdf)

        st.write("Extracted preview:")
        st.write(df_txn.head())
        if st.button("Save processed"):
            df_txn.to_csv("data/processed/model_ready.csv", index=False)
            st.success("Saved to data/processed/model_ready.csv")


# ---------------- Modeling ----------------
with tabs[2]:
    st.header("Train Risk Model (XGBoost)")
    uploaded = st.file_uploader(
        "Upload labeled CSV for risk training (must include default_flag)", type=["csv"], key="risk_train"
    )
    if uploaded:
        df_train = pd.read_csv(uploaded)
        if "default_flag" not in df_train.columns:
            st.error("Dataset must include 'default_flag' column")
        else:
            if st.button("Train risk model"):
                with st.spinner("Training..."):
                    metrics = train_risk_model(df_train, target_col="default_flag")
                    st.success(f"Trained. AUC: {metrics['auc']:.3f}")
                    st.json(metrics["report"])


# ---------------- Explainability ----------------
with tabs[3]:
    st.header("Explain predictions with SHAP")
    uploaded_explain = st.file_uploader(
        "Upload CSV of model features to explain (same features used in training)", type=["csv"], key="explain_up"
    )
    if uploaded_explain:
        df_ex = pd.read_csv(uploaded_explain)
        if Path(RISK_MODEL_PATH).exists():
            shap_vals, explainer = explain_xgb_model(df_ex, str(RISK_MODEL_PATH))
            st.write(
                "SHAP values shape:",
                None if shap_vals is None else (len(shap_vals), len(shap_vals[0]) if hasattr(shap_vals, "__len__") else None),
            )
            st.write("Feature importance (mean abs SHAP):")
            import numpy as np

            mean_abs = np.abs(shap_vals).mean(axis=0)
            feat_imp = pd.Series(mean_abs, index=df_ex.columns).sort_values(ascending=False).head(20)
            st.table(feat_imp)
        else:
            st.error("No trained risk model found. Train it first in Modeling tab.")


# ---------------- Download ----------------
with tabs[4]:
    st.header("Download artifacts")
    if Path("models").exists():
        st.write("Models folder contents:")
        import os

        st.write(os.listdir("models"))
    else:
        st.info("No models saved yet. Train and save models.")

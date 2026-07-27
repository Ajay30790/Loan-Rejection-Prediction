# app.py
# ============================================================================
# LOAN REJECTION PREDICTION & RISK ANALYTICS SYSTEM — ENHANCED
# ============================================================================
# Enhancements in this version:
#   - Robust file/model loading (checks multiple paths, no silent crashes)
#   - Graceful fallbacks when data/model files are missing (with clear guidance)
#   - Manual CSV upload fallback for the dataset
#   - Batch prediction page (upload a CSV of many applicants at once)
#   - Prediction history (session-based) with CSV export
#   - Downloadable single-prediction report
#   - Defensive column checks throughout (won't crash on schema drift)
#   - Model performance page computes its own confusion matrix / ROC curve
#     if the pre-rendered images aren't present, instead of just warning
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
from pathlib import Path
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Loan Rejection Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PATH HELPERS  (fixes the "blank dashboard" / FileNotFoundError problem)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent

def find_first_existing(candidates):
    """Return the first path (string) that exists from a list of candidates."""
    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.exists():
            return str(p)
    return None

DATA_CANDIDATES = [
    "SCFP2019.csv",
    "data/SCFP2019.csv",
    "dataset/SCFP2019.csv",
]
MODEL_CANDIDATES = [
    "model/loan_rejection_model.pkl",
    "loan_rejection_model.pkl",
]
FEATURE_IMPORTANCE_CANDIDATES = [
    "model/feature_importance.csv",
    "feature_importance.csv",
]
CONFUSION_MATRIX_IMG_CANDIDATES = ["model/visualizations/confusion_matrix.png"]
MODEL_COMPARISON_IMG_CANDIDATES = ["model/visualizations/model_comparison.png"]
ROC_CURVES_IMG_CANDIDATES = ["model/visualizations/roc_curves.png"]

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; color: #1a237e; text-align: center; padding: 1.5rem 0;
        background: linear-gradient(135deg, #e8eaf6, #c5cae9, #e8eaf6);
        border-radius: 10px; margin-bottom: 2rem; border-bottom: 4px solid #1a237e;
    }
    .header-bar {
        background: linear-gradient(90deg, #1a237e, #283593, #1a237e);
        padding: 0.8rem; border-radius: 10px; margin-bottom: 1.5rem;
        text-align: center; color: white; font-weight: bold; font-size: 1rem;
        box-shadow: 0 2px 10px rgba(26, 35, 126, 0.3);
    }
    .header-bar a {
        color: white; text-decoration: none; margin: 0 12px; padding: 4px 10px;
        border-radius: 5px; transition: background-color 0.3s; font-size: 0.9rem;
    }
    .header-bar a:hover { background-color: rgba(255, 255, 255, 0.2); text-decoration: none; }
    .header-bar .separator { color: rgba(255,255,255,0.3); margin: 0 8px; }
    .metric-card {
        background: white; padding: 1.2rem; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center;
        border-left: 4px solid #1a237e; transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
    .metric-value { font-size: 2rem; font-weight: bold; color: #1a237e; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 5px; }
    .risk-high { color: #e74c3c; font-weight: bold; padding: 2px 10px; border-radius: 4px; background: #fde8e8; display: inline-block; }
    .risk-medium { color: #f39c12; font-weight: bold; padding: 2px 10px; border-radius: 4px; background: #fef3e0; display: inline-block; }
    .risk-low { color: #2ecc71; font-weight: bold; padding: 2px 10px; border-radius: 4px; background: #e8f8ed; display: inline-block; }
    .stButton button {
        background: linear-gradient(135deg, #1a237e, #283593); color: white; font-weight: bold;
        border: none; border-radius: 8px; padding: 0.6rem 1.5rem; transition: all 0.3s; width: 100%;
    }
    .stButton button:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(26, 35, 126, 0.4); }
    .footer { text-align: center; padding: 1rem; margin-top: 2rem; border-top: 1px solid #ddd; color: #666; font-size: 0.85rem; }
    .footer a { color: #1a237e; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
    .sidebar-logo { text-align: center; padding: 0.5rem 0 1rem 0; }
    .sidebar-logo img { max-width: 60px; }
    .sidebar-title { color: #1a237e; font-size: 1.2rem; font-weight: bold; margin: 0; }
    .sidebar-subtitle { color: #666; font-size: 0.8rem; margin: 0; }
    .content-card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 1rem; }
    .badge { display: inline-block; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: bold; }
    .badge-primary { background: #e8eaf6; color: #1a237e; }
    .badge-success { background: #e8f8ed; color: #2ecc71; }
    .badge-danger { background: #fde8e8; color: #e74c3c; }
    .badge-warning { background: #fef3e0; color: #f39c12; }
    .result-approved { background: linear-gradient(135deg, #e8f8ed, #d4edda); padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid #2ecc71; }
    .result-rejected { background: linear-gradient(135deg, #fde8e8, #f8d7da); padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid #e74c3c; }
    .result-text { font-size: 2rem; font-weight: bold; }
    .result-subtext { font-size: 1.1rem; margin-top: 0.5rem; }
    .social-icons { display: flex; justify-content: center; gap: 15px; margin: 10px 0; }
    .social-icons a { display: inline-block; transition: transform 0.3s; }
    .social-icons a:hover { transform: scale(1.1); }
    .risk-bar { height: 8px; border-radius: 4px; background: #e9ecef; margin: 5px 0; overflow: hidden; }
    .risk-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
    .risk-bar-fill.high { background: #e74c3c; }
    .risk-bar-fill.medium { background: #f39c12; }
    .risk-bar-fill.low { background: #2ecc71; }
    .status-ok { color: #2ecc71; font-weight: bold; }
    .status-bad { color: #e74c3c; font-weight: bold; }

    /* ---- Sidebar navigation buttons: big, obvious, one-click ---- */
    div[data-testid="stSidebar"] div[data-testid="stButton"] button {
        text-align: left;
        justify-content: flex-start;
        background: #f0f2fa;
        color: #1a237e;
        font-weight: 600;
        font-size: 1.02rem;
        border: 1px solid #dfe3f5;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        margin-bottom: 0.4rem;
        box-shadow: none;
        transition: all 0.15s ease;
    }
    div[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: #e0e4fa;
        border-color: #1a237e;
        transform: translateX(2px);
    }
    /* Active page = Streamlit "primary" button type */
    div[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white;
        border-color: #1a237e;
        box-shadow: 0 2px 10px rgba(26, 35, 126, 0.35);
    }
    div[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: none;
        background: linear-gradient(135deg, #1a237e, #283593);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INIT
# ============================================================================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "uploaded_data_override" not in st.session_state:
    st.session_state.uploaded_data_override = None

# ============================================================================
# LOAD DATA AND MODEL — ROBUST VERSIONS
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data_from_path(path):
    return pd.read_csv(path)

def load_data():
    """
    Tries known paths first. If none are found, offers the user a manual
    upload widget instead of silently returning None / crashing the app.
    """
    if st.session_state.uploaded_data_override is not None:
        return st.session_state.uploaded_data_override

    path = find_first_existing(DATA_CANDIDATES)
    if path:
        try:
            return load_data_from_path(path)
        except Exception as e:
            st.error(f"❌ Found a data file at `{path}` but couldn't read it: {e}")
            return None

    # Not found anywhere — don't just fail silently.
    st.warning(
        "⚠️ Couldn't find **SCFP2019.csv** in the app folder, `data/`, or `dataset/`. "
        "Upload it below to continue, or add it to your repo so this warning "
        "stops appearing."
    )
    uploaded = st.file_uploader("Upload SCFP2019.csv", type=["csv"], key="data_upload_fallback")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.uploaded_data_override = df
            st.success("✅ Data loaded from upload. Reloading...")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_model():
    path = find_first_existing(MODEL_CANDIDATES)
    if not path:
        return None
    try:
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.session_state["_model_load_error"] = str(e)
        return None

@st.cache_data(show_spinner=False)
def create_risk_features(data):
    """Create risk indicators and target variable — defensive to missing cols."""
    data = data.copy()

    def col(name, default=0):
        if name not in data.columns:
            data[name] = default
        return data[name]

    col('DEBT2INC'); col('LEVRATIO'); col('SAVED'); col('EMERGSAV')
    col('TURNDOWN'); col('FEARDENIAL'); col('BNKRUPLAST5')
    col('FORECLLAST5'); col('LATE60'); col('HPAYDAY')

    data['DEBT2INC_RISK'] = (data['DEBT2INC'] > 0.4).astype(int)
    data['LEVRATIO_RISK'] = (data['LEVRATIO'] > 0.5).astype(int)
    data['LOW_SAVINGS'] = (data['SAVED'] == 0).astype(int)
    data['NO_EMERGENCY_SAV'] = (data['EMERGSAV'] == 0).astype(int)

    data['LOAN_REJECTED'] = 0
    for flag_col in ['TURNDOWN', 'FEARDENIAL', 'BNKRUPLAST5', 'FORECLLAST5', 'LATE60', 'HPAYDAY']:
        data.loc[data[flag_col] == 1, 'LOAN_REJECTED'] = 1

    data['RISK_SCORE'] = (
        data['DEBT2INC_RISK'] * 2 +
        data['LEVRATIO_RISK'] * 2 +
        data['LOW_SAVINGS'] * 1.5 +
        data['NO_EMERGENCY_SAV'] * 1.5
    )
    return data

def safe_col(data, name, fallback=0.0):
    """Return a column if present, else a Series of the fallback value."""
    if name in data.columns:
        return data[name]
    return pd.Series([fallback] * len(data), index=data.index)

# ============================================================================
# EXPLAINABLE AI (XAI) HELPERS
# ----------------------------------------------------------------------------
# These are model-agnostic (work for logistic regression, random forest,
# gradient boosting, or anything with predict_proba) so we don't need to add
# a heavy dependency like the `shap` package, which can be slow/fragile to
# install on Streamlit Community Cloud.
#
# Local explanation = "ablation"/occlusion method: for one applicant, swap
# each feature one at a time for the population's typical (mean) value and
# see how much the predicted rejection probability moves. A feature that
# moves the probability a lot is an important driver for THIS applicant.
#
# Global explanation fallback = sklearn's permutation_importance, used only
# when the model doesn't natively expose feature_importances_ / coef_.
# ============================================================================

def explain_single_prediction(model, scaler, feature_names, input_dict, background_df, top_n=10):
    """
    Returns (baseline_probability, DataFrame of feature contributions) for a
    single applicant, using an ablation / occlusion approach.
    Positive contribution = this feature's actual value PUSHES TOWARD rejection.
    Negative contribution = this feature's actual value PUSHES TOWARD approval.
    """
    avail_feats = [f for f in feature_names if f in background_df.columns]
    if not avail_feats:
        avail_feats = feature_names

    bg = background_df.reindex(columns=feature_names, fill_value=0).fillna(0)
    if len(bg) > 300:
        bg = bg.sample(300, random_state=42)
    typical_values = bg.mean()

    input_row = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0).fillna(0)
    baseline_scaled = scaler.transform(input_row)
    baseline_proba = float(model.predict_proba(baseline_scaled)[0][1])

    contributions = []
    for feat in feature_names:
        modified_row = input_row.copy()
        modified_row[feat] = typical_values[feat]
        modified_scaled = scaler.transform(modified_row)
        modified_proba = float(model.predict_proba(modified_scaled)[0][1])
        delta = baseline_proba - modified_proba
        contributions.append({
            "Feature": feat,
            "Applicant_Value": input_dict.get(feat, 0),
            "Typical_Value": round(float(typical_values[feat]), 3),
            "Contribution": delta
        })

    contrib_df = pd.DataFrame(contributions)
    contrib_df["Abs_Contribution"] = contrib_df["Contribution"].abs()
    contrib_df = contrib_df.sort_values("Abs_Contribution", ascending=False).head(top_n)
    return baseline_proba, contrib_df

def plot_contribution_chart(contrib_df, title="Why this prediction?"):
    """Horizontal bar chart: red = pushes toward rejection, green = pushes toward approval."""
    plot_df = contrib_df.sort_values("Contribution")
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in plot_df["Contribution"]]
    fig = go.Figure(go.Bar(
        x=plot_df["Contribution"] * 100,
        y=plot_df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v*100:+.1f} pp" for v in plot_df["Contribution"]],
        textposition="outside"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Change in rejection probability (percentage points)",
        yaxis_title="",
        height=max(350, 35 * len(plot_df)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=50, b=40)
    )
    return fig

def get_global_feature_importance(model, scaler, feature_names, data, target_col='LOAN_REJECTED', sample_size=500):
    """
    Universal fallback for global feature importance:
    1. Native feature_importances_ (tree models)
    2. Native coef_ (linear models)
    3. sklearn permutation_importance (anything else, e.g. SVM/KNN)
    """
    if hasattr(model, 'feature_importances_'):
        return pd.DataFrame({
            'Feature': feature_names, 'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False), "native (feature_importances_)"

    if hasattr(model, 'coef_'):
        coefs = model.coef_[0] if np.ndim(model.coef_) > 1 else model.coef_
        return pd.DataFrame({
            'Feature': feature_names, 'Importance': np.abs(coefs)
        }).sort_values('Importance', ascending=False), "native (coefficients, absolute value)"

    if data is not None and target_col in data.columns:
        avail = [f for f in feature_names if f in data.columns]
        X = data[avail].fillna(0)
        y = data[target_col]
        if len(X) > sample_size:
            idx = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]
        X_scaled = scaler.transform(X)
        perm = permutation_importance(model, X_scaled, y, n_repeats=5, random_state=42, n_jobs=-1)
        return pd.DataFrame({
            'Feature': avail, 'Importance': perm.importances_mean
        }).sort_values('Importance', ascending=False), "permutation importance (model-agnostic)"

    return None, None

# Load data and model
data = load_data()
if data is not None:
    data = create_risk_features(data)
model_data = load_model()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <img src="https://img.icons8.com/color/96/000000/university.png" style="width: 60px;">
        <p class="sidebar-title">Loan Rejection Predictor</p>
        <p class="sidebar-subtitle">Risk Analytics System</p>
        <hr style="margin: 10px 0;">
        <p style="font-size: 0.8rem; color: #1a237e; font-weight: bold;">📌 Advanced Analytics</p>
        <p style="font-size: 0.8rem; color: #666;">Predictive Modeling & XAI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    pages = [
        "🏠 Dashboard",
        "📊 Dataset Overview",
        "📈 Risk Analytics",
        "🤖 Model Performance",
        "🎯 Predict Loan Status",
        "📥 Batch Prediction",
        "🕑 Prediction History",
        "📚 Documentation"
    ]

    if "nav_page" not in st.session_state:
        st.session_state.nav_page = pages[0]

    st.markdown("#### 🧭 Navigate to")
    for p in pages:
        is_active = (st.session_state.nav_page == p)
        if st.button(
            p,
            key=f"nav_btn_{p}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.nav_page = p
            st.rerun()

    page = st.session_state.nav_page

    st.markdown("---")
    st.markdown("#### 🔧 System Status")
    st.markdown(
        f"Data: {'<span class=\"status-ok\">✅ Loaded</span>' if data is not None else '<span class=\"status-bad\">❌ Missing</span>'}",
        unsafe_allow_html=True
    )
    st.markdown(
        f"Model: {'<span class=\"status-ok\">✅ Loaded</span>' if model_data is not None else '<span class=\"status-bad\">❌ Missing</span>'}",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #666; text-align: center;">
        <p style="margin: 0;">🏛️ Loan Analytics System</p>
        <hr style="margin: 10px 0;">
        <div class="social-icons">
            <a href="https://github.com/Ajay30790" target="_blank"><img src="https://img.icons8.com/ios-glyphs/25/000000/github.png" style="width: 22px;"></a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="header-bar">
    <span>🏛️ Loan Rejection Prediction & Risk Analytics</span>
    <span class="separator">|</span>
    <span>📊 Advanced Analytics</span>
    <span class="separator">|</span>
    <span>🤖 Machine Learning</span>
    <span class="separator">|</span>
    <a href="https://github.com/Ajay30790" target="_blank">🐙 GitHub</a>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Loan Rejection Prediction & Risk Analytics System</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown("### 📊 Dashboard Overview")
    st.markdown("---")

    if data is None:
        st.info("👆 Load a dataset (see the upload box above, or place SCFP2019.csv in the app folder) to see the dashboard.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(data):,}</div><div class="metric-label">📋 Total Applicants</div></div>', unsafe_allow_html=True)
        with col2:
            rejected = int(data['LOAN_REJECTED'].sum())
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#e74c3c;">{rejected:,}</div><div class="metric-label">❌ Rejected Applications</div></div>', unsafe_allow_html=True)
        with col3:
            rate = (rejected / len(data)) * 100 if len(data) else 0
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f39c12;">{rate:.1f}%</div><div class="metric-label">📊 Rejection Rate</div></div>', unsafe_allow_html=True)
        with col4:
            avg_income = safe_col(data, 'INCOME').mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">₹{avg_income:,.0f}</div><div class="metric-label">💰 Average Income</div></div>', unsafe_allow_html=True)
        with col5:
            avg_debt = safe_col(data, 'DEBT').mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">₹{avg_debt:,.0f}</div><div class="metric-label">💳 Average Debt</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Rejection Rate by Risk Category")
            risk_categories = {
                'High DTI (>0.4)': data[data['DEBT2INC_RISK'] == 1]['LOAN_REJECTED'].mean() * 100 if (data['DEBT2INC_RISK'] == 1).any() else 0,
                'High Leverage (>0.5)': data[data['LEVRATIO_RISK'] == 1]['LOAN_REJECTED'].mean() * 100 if (data['LEVRATIO_RISK'] == 1).any() else 0,
                'Low Savings': data[data['LOW_SAVINGS'] == 1]['LOAN_REJECTED'].mean() * 100 if (data['LOW_SAVINGS'] == 1).any() else 0,
                'No Emergency Savings': data[data['NO_EMERGENCY_SAV'] == 1]['LOAN_REJECTED'].mean() * 100 if (data['NO_EMERGENCY_SAV'] == 1).any() else 0,
            }
            fig = px.bar(
                x=list(risk_categories.keys()), y=list(risk_categories.values()),
                title="Rejection Rate by Risk Category",
                labels={'x': 'Risk Category', 'y': 'Rejection Rate (%)'},
                color=list(risk_categories.values()), color_continuous_scale='Reds', text_auto='.1f'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12), height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 💰 Income vs Debt Distribution")
            if 'INCOME' in data.columns and 'DEBT' in data.columns:
                sample_df = data.sample(min(2000, len(data)), random_state=42)
                fig = px.scatter(
                    sample_df, x='INCOME', y='DEBT', color='LOAN_REJECTED',
                    title="Income vs Debt by Loan Status",
                    labels={'INCOME': 'Income (₹)', 'DEBT': 'Debt (₹)'},
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, opacity=0.6
                )
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12), height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("INCOME/DEBT columns not found in this dataset.")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Rejection Rate by Age Group")
            if 'AGE' in data.columns:
                age_bins = pd.cut(data['AGE'], bins=[20, 30, 40, 50, 60, 70, 80, 100])
                age_rejection = data.groupby(age_bins, observed=True)['LOAN_REJECTED'].mean() * 100
                fig = px.bar(
                    x=[f'{int(b.left)}-{int(b.right)}' for b in age_rejection.index],
                    y=age_rejection.values, title="Rejection Rate by Age Group",
                    labels={'x': 'Age Group', 'y': 'Rejection Rate (%)'},
                    color=age_rejection.values, color_continuous_scale='Blues'
                )
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12), height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("AGE column not found in this dataset.")

        with col2:
            st.markdown("### 🎓 Rejection Rate by Education Level")
            if 'EDCL' in data.columns:
                edu_rejection = data.groupby('EDCL')['LOAN_REJECTED'].mean() * 100
                fig = px.bar(
                    x=edu_rejection.index.astype(str), y=edu_rejection.values,
                    title="Rejection Rate by Education Level",
                    labels={'x': 'Education Level', 'y': 'Rejection Rate (%)'},
                    color=edu_rejection.values, color_continuous_scale='Greens'
                )
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12), height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("EDCL column not found in this dataset.")

        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            hi = data[data['RISK_SCORE'] >= 4]
            high_risk_rate = hi['LOAN_REJECTED'].mean() * 100 if len(hi) else 0
            st.metric("High Risk Applicants", f"{high_risk_rate:.1f}%", "Rejection Rate", delta_color="inverse")
        with col2:
            lo = data[data['RISK_SCORE'] <= 1]
            low_risk_rate = lo['LOAN_REJECTED'].mean() * 100 if len(lo) else 0
            st.metric("Low Risk Applicants", f"{low_risk_rate:.1f}%", "Rejection Rate", delta_color="normal")
        with col3:
            rej = data[data['LOAN_REJECTED'] == 1]
            avg_dti_rejected = rej['DEBT2INC'].mean() if len(rej) else 0
            st.metric("Avg DTI (Rejected)", f"{avg_dti_rejected:.3f}", "Debt-to-Income Ratio")

        st.markdown("---")
        csv_export = data.describe().to_csv().encode('utf-8')
        st.download_button("⬇️ Download Summary Statistics (CSV)", csv_export, "dashboard_summary_stats.csv", "text/csv")

# ============================================================================
# PAGE 2: DATASET OVERVIEW
# ============================================================================
elif page == "📊 Dataset Overview":
    st.markdown("### 📊 Dataset Overview")
    st.markdown("---")

    if data is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📋 Total Rows", f"{len(data):,}")
        with col2:
            st.metric("📊 Total Columns", f"{len(data.columns):,}")
        with col3:
            st.metric("💾 Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        st.markdown("---")
        st.markdown("### 📄 Data Preview")
        st.dataframe(data.head(100), use_container_width=True)

        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.values,
                'Non-Null Count': data.notnull().sum().values,
                'Null Percentage': (data.isnull().sum() / len(data) * 100).values
            })
            st.dataframe(col_info, use_container_width=True)

        with st.expander("📊 Summary Statistics"):
            st.dataframe(data.describe(), use_container_width=True)

        st.markdown("### 🔍 Missing Values Analysis")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values, y=missing_data.index, title="Missing Values by Column",
                labels={'x': 'Missing Count', 'y': 'Column'},
                color=missing_data.values, color_continuous_scale='Reds', height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")

        st.markdown("---")
        st.download_button(
            "⬇️ Download Current Dataset (CSV)",
            data.to_csv(index=False).encode('utf-8'),
            "dataset_export.csv", "text/csv"
        )
    else:
        st.info("No dataset loaded yet — use the upload box at the top of the page.")

# ============================================================================
# PAGE 3: RISK ANALYTICS
# ============================================================================
elif page == "📈 Risk Analytics":
    st.markdown("### 📈 Risk Analytics Dashboard")
    st.markdown("---")

    if data is not None:

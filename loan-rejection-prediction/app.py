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
        st.markdown("### 🎯 Risk Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            high_dti = (data['DEBT2INC'] > 0.4).mean() * 100
            rr = data[data['DEBT2INC_RISK'] == 1]['LOAN_REJECTED'].mean() * 100 if (data['DEBT2INC_RISK'] == 1).any() else 0
            st.metric("📊 High DTI (>0.4)", f"{high_dti:.1f}%", delta=f"{rr:.1f}% rejection")
        with col2:
            high_lev = (data['LEVRATIO'] > 0.5).mean() * 100
            rr = data[data['LEVRATIO_RISK'] == 1]['LOAN_REJECTED'].mean() * 100 if (data['LEVRATIO_RISK'] == 1).any() else 0
            st.metric("📊 High Leverage (>0.5)", f"{high_lev:.1f}%", delta=f"{rr:.1f}% rejection")
        with col3:
            low_savings = (data['SAVED'] == 0).mean() * 100
            rr = data[data['LOW_SAVINGS'] == 1]['LOAN_REJECTED'].mean() * 100 if (data['LOW_SAVINGS'] == 1).any() else 0
            st.metric("📊 Low/No Savings", f"{low_savings:.1f}%", delta=f"{rr:.1f}% rejection")
        with col4:
            no_emerg = (data['EMERGSAV'] == 0).mean() * 100
            rr = data[data['NO_EMERGENCY_SAV'] == 1]['LOAN_REJECTED'].mean() * 100 if (data['NO_EMERGENCY_SAV'] == 1).any() else 0
            st.metric("📊 No Emergency Savings", f"{no_emerg:.1f}%", delta=f"{rr:.1f}% rejection")

        st.markdown("---")
        st.markdown("### 🔗 Risk Indicator Correlation Matrix")
        risk_cols = ['DEBT2INC', 'LEVRATIO', 'LOAN_REJECTED', 'DEBT2INC_RISK', 'LEVRATIO_RISK', 'LOW_SAVINGS', 'NO_EMERGENCY_SAV']
        risk_cols_available = [c for c in risk_cols if c in data.columns]
        corr_matrix = data[risk_cols_available].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                         title="Correlation Matrix of Risk Indicators", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Risk Distribution Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(data, names='LOAN_REJECTED', title='Loan Rejection Distribution',
                         color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.3, height=400)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(data, x='DEBT2INC', color='LOAN_REJECTED', title='Debt-to-Income Ratio Distribution',
                                labels={'DEBT2INC': 'Debt-to-Income Ratio', 'count': 'Count'},
                                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, nbins=50, height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Advanced Risk Analytics")
        col1, col2 = st.columns(2)
        with col1:
            if 'INCOME' in data.columns:
                fig = px.box(data, x='LOAN_REJECTED', y='INCOME', title='Income Distribution by Loan Status',
                             labels={'LOAN_REJECTED': 'Loan Rejected', 'INCOME': 'Income (₹)'},
                             color='LOAN_REJECTED', color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, height=400)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'NETWORTH' in data.columns:
                fig = px.box(data, x='LOAN_REJECTED', y='NETWORTH', title='Net Worth Distribution by Loan Status',
                             labels={'LOAN_REJECTED': 'Loan Rejected', 'NETWORTH': 'Net Worth (₹)'},
                             color='LOAN_REJECTED', color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, height=400)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🎯 Composite Risk Score Distribution")
        fig = px.histogram(data, x='RISK_SCORE', color='LOAN_REJECTED', title='Composite Risk Score Distribution',
                            labels={'RISK_SCORE': 'Risk Score', 'count': 'Count'},
                            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, barmode='stack', height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Risk Score Breakdown")
        risk_summary = data.groupby('RISK_SCORE').agg({'LOAN_REJECTED': ['count', 'mean']}).reset_index()
        risk_summary.columns = ['Risk Score', 'Count', 'Rejection Rate']
        risk_summary['Rejection Rate'] = risk_summary['Rejection Rate'] * 100
        risk_summary['Risk Level'] = risk_summary['Risk Score'].apply(
            lambda x: 'High' if x >= 4 else 'Medium' if x >= 2 else 'Low'
        )
        st.dataframe(risk_summary, use_container_width=True)
    else:
        st.info("No dataset loaded yet — go to Dashboard or Dataset Overview to upload one.")

# ============================================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================================
elif page == "🤖 Model Performance":
    st.markdown("### 🤖 Model Performance Analysis")
    st.markdown("---")

    if model_data is None:
        err = st.session_state.get("_model_load_error")
        st.warning("⚠️ No trained model found at `model/loan_rejection_model.pkl`.")
        if err:
            st.caption(f"Last load error: {err}")
        st.info("Train a model and save it there, or upload one below.")
        uploaded_model = st.file_uploader("Upload model .pkl", type=["pkl"], key="model_upload")
        if uploaded_model is not None:
            try:
                model_data = pickle.load(uploaded_model)
                st.success("✅ Model loaded from upload for this session.")
            except Exception as e:
                st.error(f"❌ Couldn't load uploaded model: {e}")

    if model_data is not None and data is not None:
        st.markdown("### 📊 Model Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Model", model_data.get('model_name', 'N/A'))
        with col2:
            performance = model_data.get('performance', {})
            st.metric("📈 ROC-AUC Score", f"{performance.get('roc_auc', 0):.4f}")
        with col3:
            st.metric("🎯 Accuracy", f"{performance.get('accuracy', 0):.4f}")

        st.markdown("---")
        st.markdown("### 🔍 Feature Importance Analysis (Global XAI)")
        fi_path = find_first_existing(FEATURE_IMPORTANCE_CANDIDATES)
        feature_importance = None
        importance_source = None
        if fi_path:
            try:
                feature_importance = pd.read_csv(fi_path)
                importance_source = "pre-computed file"
            except Exception as e:
                st.warning(f"⚠️ Couldn't read feature importance file: {e}")

        if feature_importance is None and 'model' in model_data:
            try:
                model = model_data['model']
                scaler = model_data['scaler']
                feat_names = model_data.get('feature_names', [])
                if feat_names:
                    with st.spinner("Computing feature importance..."):
                        feature_importance, importance_source = get_global_feature_importance(
                            model, scaler, feat_names, data
                        )
            except Exception as e:
                st.warning(f"⚠️ Couldn't compute feature importance: {e}")

        if feature_importance is not None:
            st.caption(f"Source: {importance_source}")
            fig = px.bar(feature_importance.head(15), x='Importance', y='Feature',
                         title='Top 15 Most Important Features', orientation='h',
                         color='Importance', color_continuous_scale='Viridis', height=500)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Feature importance not available (no file found, and the fallback methods failed).")

        st.markdown("### 📈 Model Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            values = [performance.get('accuracy', 0), performance.get('precision', 0),
                      performance.get('recall', 0), performance.get('f1', 0),
                      performance.get('roc_auc', 0)]
            fig = px.bar(x=metrics, y=values, title='Performance Metrics',
                         labels={'x': 'Metric', 'y': 'Score'}, color=values,
                         color_continuous_scale='Blues', range_y=[0, 1], height=400)
            fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Confusion Matrix")
            cm_img = find_first_existing(CONFUSION_MATRIX_IMG_CANDIDATES)
            if cm_img:
                st.image(cm_img, use_container_width=True)
            else:
                # Compute one live instead of just warning
                try:
                    model = model_data['model']
                    scaler = model_data['scaler']
                    features = model_data['feature_names']
                    available_feat = [f for f in features if f in data.columns]
                    X = data[available_feat].fillna(0)
                    X_scaled = scaler.transform(X)
                    y_pred = model.predict(X_scaled)
                    cm = confusion_matrix(data['LOAN_REJECTED'], y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                       labels=dict(x="Predicted", y="Actual"),
                                       x=['Approved', 'Rejected'], y=['Approved', 'Rejected'],
                                       title="Confusion Matrix (computed live)")
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Couldn't compute a confusion matrix: {e}")

        st.markdown("### 📊 Model Comparison")
        mc_img = find_first_existing(MODEL_COMPARISON_IMG_CANDIDATES)
        if mc_img:
            st.image(mc_img, use_container_width=True)
        else:
            st.info("ℹ️ No pre-rendered model comparison image found.")

        st.markdown("### 📈 ROC Curve")
        roc_img = find_first_existing(ROC_CURVES_IMG_CANDIDATES)
        if roc_img:
            st.image(roc_img, use_container_width=True)
        else:
            try:
                model = model_data['model']
                scaler = model_data['scaler']
                features = model_data['feature_names']
                available_feat = [f for f in features if f in data.columns]
                X = data[available_feat].fillna(0)
                X_scaled = scaler.transform(X)
                y_proba = model.predict_proba(X_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(data['LOAN_REJECTED'], y_proba)
                roc_auc_val = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc_val:.3f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                fig_roc.update_layout(title="ROC Curve (computed live)", xaxis_title="False Positive Rate",
                                       yaxis_title="True Positive Rate", height=450)
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as e:
                st.info(f"ℹ️ No ROC image found and couldn't compute one live: {e}")

        st.markdown("### 📋 Detailed Classification Report")
        try:
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['feature_names']
            available_feat = [f for f in features if f in data.columns]
            X = data[available_feat].fillna(0)
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            report = classification_report(data['LOAN_REJECTED'], y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Unable to generate classification report: {str(e)}")
    elif model_data is not None and data is None:
        st.info("Model is loaded, but no dataset is available to evaluate it against.")

# ============================================================================
# PAGE 5: PREDICT LOAN STATUS (single applicant)
# ============================================================================
elif page == "🎯 Predict Loan Status":
    st.markdown("### 🎯 Predict Loan Rejection Status")
    st.markdown("---")

    if model_data is None:
        st.warning("⚠️ No model loaded. Go to **Model Performance** to upload one, or add it to `model/loan_rejection_model.pkl`.")
    else:
        st.info("💡 Enter the applicant's financial information to predict loan rejection risk")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Personal Information")
            age = st.slider("Age", 18, 100, 35, key="age")
            education = st.selectbox("Education Level", options=[1, 2, 3, 4, 5],
                format_func=lambda x: {1: "Less than High School", 2: "High School",
                                        3: "Some College", 4: "Bachelor's", 5: "Advanced"}.get(x, str(x)), key="education")
            married = st.selectbox("Marital Status", options=[1, 2],
                format_func=lambda x: "Married" if x == 1 else "Not Married", key="married")
            kids = st.selectbox("Has Children", options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: f"{x} children", key="kids")

        with col2:
            st.markdown("#### 💰 Financial Information")
            income = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=10000, key="income")
            debt = st.number_input("Total Debt (₹)", min_value=0, value=250000, step=10000, key="debt")
            assets = st.number_input("Total Assets (₹)", min_value=0, value=1000000, step=50000, key="assets")
            networth = st.number_input("Net Worth (₹)", min_value=0, value=500000, step=50000, key="networth")

        with col3:
            st.markdown("#### 📊 Risk Indicators")
            savings = st.selectbox("Has Savings", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="savings")
            emergency_sav = st.selectbox("Has Emergency Savings", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="emergency_sav")
            credit_turn_down = st.selectbox("Previously Turned Down for Credit", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="credit_turn_down")
            late_payment = st.selectbox("Late on Payments (60+ days)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="late_payment")

        if income == 0:
            st.warning("⚠️ Income is ₹0 — debt-to-income ratio will be undefined/treated as 0. Consider entering a realistic income.")
        if assets == 0:
            st.warning("⚠️ Assets are ₹0 — leverage ratio will be undefined/treated as 0.")

        debt_to_income = debt / income if income > 0 else 0
        leverage = debt / assets if assets > 0 else 0
        high_dti = 1 if debt_to_income > 0.4 else 0
        high_leverage = 1 if leverage > 0.5 else 0
        low_savings = 1 if savings == 0 else 0
        no_emergency_sav = 1 if emergency_sav == 0 else 0
        risk_score = high_dti + high_leverage + low_savings + no_emergency_sav

        features = {
            'AGE': age, 'INCOME': income, 'DEBT': debt, 'ASSET': assets, 'NETWORTH': networth,
            'EDUC': education, 'MARRIED': married, 'KIDS': kids, 'SAVED': savings,
            'EMERGSAV': emergency_sav, 'TURNDOWN': credit_turn_down, 'LATE60': late_payment,
            'DEBT2INC': debt_to_income, 'LEVRATIO': leverage, 'DEBT2INC_RISK': high_dti,
            'LEVRATIO_RISK': high_leverage, 'LOW_SAVINGS': low_savings, 'NO_EMERGENCY_SAV': no_emergency_sav,
        }
        model_features = model_data['feature_names']
        for feat in model_features:
            if feat not in features:
                features[feat] = 0

        st.markdown("---")

        if st.button("🔮 Predict Loan Rejection Risk", use_container_width=True, key="predict_button"):
            try:
                X_pred = pd.DataFrame([features])[model_features].fillna(0)
                scaler = model_data['scaler']
                model = model_data['model']
                X_scaled = scaler.transform(X_pred)
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1]
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                prediction, probability = None, None

            if prediction is not None:
                st.markdown("## 📊 Prediction Results")
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if prediction == 0:
                        st.markdown(f"""
                        <div class="result-approved">
                            <div class="result-text" style="color: #2ecc71;">✅ Loan Approved</div>
                            <div class="result-subtext">Rejection Probability: {(1 - probability) * 100:.1f}%</div>
                            <span class="badge badge-success">Low Risk</span>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-rejected">
                            <div class="result-text" style="color: #e74c3c;">❌ Loan Rejected</div>
                            <div class="result-subtext">Rejection Probability: {probability * 100:.1f}%</div>
                            <span class="badge badge-danger">High Risk</span>
                        </div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("## 🎯 Risk Assessment")
                risk_factors = []
                if high_dti: risk_factors.append(("High Debt-to-Income Ratio", "High"))
                if high_leverage: risk_factors.append(("High Leverage Ratio", "High"))
                if low_savings: risk_factors.append(("Low/No Savings", "Medium"))
                if no_emergency_sav: risk_factors.append(("No Emergency Savings", "Medium"))
                if late_payment: risk_factors.append(("Late Payments", "High"))
                if credit_turn_down: risk_factors.append(("Previous Credit Denial", "High"))

                if risk_factors:
                    col1, col2 = st.columns(2)
                    for i, (factor, level) in enumerate(risk_factors):
                        with col1 if i % 2 == 0 else col2:
                            level_class = "risk-high" if level == "High" else "risk-medium"
                            st.markdown(f"<span class='{level_class}'>{level}</span> {factor}", unsafe_allow_html=True)
                else:
                    st.success("✅ No significant risk factors identified")

                st.markdown("---")
                st.markdown("## 💰 Financial Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    dti_status = "🔴 High" if debt_to_income > 0.4 else "🟢 Acceptable"
                    st.metric("Debt-to-Income Ratio", f"{debt_to_income:.3f}", delta=dti_status)
                with col2:
                    lev_status = "🔴 High" if leverage > 0.5 else "🟢 Acceptable"
                    st.metric("Leverage Ratio", f"{leverage:.3f}", delta=lev_status)
                with col3:
                    risk_level = "🔴 High" if risk_score >= 3 else "🟡 Medium" if risk_score >= 2 else "🟢 Low"
                    st.metric("Risk Score", f"{risk_score}/4", delta=risk_level)

                st.markdown("### 📊 Risk Meter")
                risk_percentage = (risk_score / 4) * 100
                color = "high" if risk_percentage >= 75 else "medium" if risk_percentage >= 50 else "low"
                st.markdown(f"""
                <div class="risk-bar"><div class="risk-bar-fill {color}" style="width: {risk_percentage}%;"></div></div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666;">
                    <span>Low Risk</span><span>{risk_percentage:.0f}%</span><span>High Risk</span>
                </div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("## 🔍 Why This Prediction? (Explainable AI)")
                if data is not None:
                    try:
                        with st.spinner("Computing explanation..."):
                            _, contrib_df = explain_single_prediction(
                                model, scaler, model_features, features, data, top_n=10
                            )
                        st.caption(
                            "Each bar shows how much this applicant's actual value for that "
                            "feature changed the rejection probability, compared to a typical "
                            "applicant in the dataset. 🔴 Red = pushed toward rejection. "
                            "🟢 Green = pushed toward approval."
                        )
                        fig_xai = plot_contribution_chart(contrib_df, title="Top factors driving this prediction")
                        st.plotly_chart(fig_xai, use_container_width=True)

                        with st.expander("📋 See exact values behind this explanation"):
                            display_df = contrib_df[["Feature", "Applicant_Value", "Typical_Value", "Contribution"]].copy()
                            display_df["Contribution (pp)"] = (display_df["Contribution"] * 100).round(2)
                            display_df = display_df.drop(columns=["Contribution"])
                            st.dataframe(display_df, use_container_width=True)

                        st.download_button(
                            "⬇️ Download Explanation (CSV)",
                            contrib_df.to_csv(index=False).encode('utf-8'),
                            file_name="prediction_explanation.csv"
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Couldn't generate an explanation: {e}")
                else:
                    st.info("ℹ️ Load a dataset (Dashboard or Dataset Overview) to enable explanations — they're computed by comparing this applicant against typical values in your data.")

                # Save to session history
                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "age": age, "income": income, "debt": debt, "assets": assets,
                    "networth": networth, "debt_to_income": round(debt_to_income, 3),
                    "leverage": round(leverage, 3), "risk_score": risk_score,
                    "prediction": "Rejected" if prediction == 1 else "Approved",
                    "rejection_probability": round(float(probability) * 100, 2)
                }
                st.session_state.prediction_history.append(record)

                # Downloadable single report
                report_text = io.StringIO()
                report_text.write("LOAN REJECTION PREDICTION REPORT\n")
                report_text.write("=" * 40 + "\n")
                for k, v in record.items():
                    report_text.write(f"{k}: {v}\n")
                st.download_button("⬇️ Download This Report (TXT)", report_text.getvalue(),
                                    file_name=f"loan_report_{record['timestamp'].replace(':', '-')}.txt")

# ============================================================================
# PAGE 6: BATCH PREDICTION
# ============================================================================
elif page == "📥 Batch Prediction":
    st.markdown("### 📥 Batch Prediction")
    st.markdown("---")

    if model_data is None:
        st.warning("⚠️ No model loaded. Go to **Model Performance** to upload one first.")
    else:
        st.info(
            "Upload a CSV where each row is an applicant. Missing columns required by the "
            "model will be filled with 0 automatically."
        )
        model_features = model_data['feature_names']
        st.caption(f"Model expects these features: {', '.join(model_features)}")

        batch_file = st.file_uploader("Upload applicants CSV", type=["csv"], key="batch_upload")
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                st.markdown("#### Preview")
                st.dataframe(batch_df.head(20), use_container_width=True)

                if st.button("🔮 Run Batch Prediction", use_container_width=True):
                    X_batch = batch_df.copy()
                    for feat in model_features:
                        if feat not in X_batch.columns:
                            X_batch[feat] = 0
                    X_batch = X_batch[model_features].fillna(0)

                    scaler = model_data['scaler']
                    model = model_data['model']
                    X_scaled = scaler.transform(X_batch)
                    preds = model.predict(X_scaled)
                    probs = model.predict_proba(X_scaled)[:, 1]

                    results = batch_df.copy()
                    results['Predicted_Status'] = np.where(preds == 1, 'Rejected', 'Approved')
                    results['Rejection_Probability_%'] = (probs * 100).round(2)

                    # Persist so the row-level explanation picker below survives reruns
                    st.session_state.batch_results = results
                    st.session_state.batch_X = X_batch

                if "batch_results" in st.session_state:
                    results = st.session_state.batch_results
                    X_batch = st.session_state.batch_X

                    st.markdown("#### Results")
                    st.dataframe(results, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Applicants", f"{len(results):,}")
                    with col2:
                        n_rejected = int((results['Predicted_Status'] == 'Rejected').sum())
                        st.metric("Predicted Rejections", f"{n_rejected:,}")
                    with col3:
                        st.metric("Predicted Rejection Rate", f"{(n_rejected/len(results)*100):.1f}%")

                    fig = px.histogram(results, x='Rejection_Probability_%', nbins=30,
                                        title="Distribution of Predicted Rejection Probability")
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        "⬇️ Download Predictions (CSV)",
                        results.to_csv(index=False).encode('utf-8'),
                        "batch_predictions.csv", "text/csv"
                    )

                    st.markdown("---")
                    st.markdown("### 🔍 Explain an Individual Applicant (XAI)")
                    if data is not None:
                        row_idx = st.selectbox(
                            "Pick a row to explain",
                            options=list(results.index),
                            format_func=lambda i: f"Row {i} — {results.loc[i, 'Predicted_Status']} ({results.loc[i, 'Rejection_Probability_%']}%)",
                            key="batch_explain_row"
                        )
                        if st.button("🔍 Explain This Applicant", key="explain_batch_row_btn"):
                            try:
                                scaler = model_data['scaler']
                                model = model_data['model']
                                input_dict = X_batch.loc[row_idx].to_dict()
                                with st.spinner("Computing explanation..."):
                                    _, contrib_df = explain_single_prediction(
                                        model, scaler, model_features, input_dict, data, top_n=10
                                    )
                                st.caption(
                                    "🔴 Red = pushed this applicant toward rejection. "
                                    "🟢 Green = pushed toward approval, relative to a typical applicant."
                                )
                                fig_xai = plot_contribution_chart(contrib_df, title=f"Why row {row_idx} was predicted this way")
                                st.plotly_chart(fig_xai, use_container_width=True)
                            except Exception as e:
                                st.warning(f"⚠️ Couldn't generate an explanation for this row: {e}")
                    else:
                        st.info("ℹ️ Load a dataset (Dashboard or Dataset Overview) to enable per-applicant explanations.")
            except Exception as e:
                st.error(f"❌ Error processing batch file: {e}")

# ============================================================================
# PAGE 7: PREDICTION HISTORY
# ============================================================================
elif page == "🕑 Prediction History":
    st.markdown("### 🕑 Prediction History (this session)")
    st.markdown("---")

    history = st.session_state.prediction_history
    if not history:
        st.info("No predictions made yet in this session. Go to **Predict Loan Status** to run one.")
    else:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", f"{len(hist_df):,}")
        with col2:
            n_rej = (hist_df['prediction'] == 'Rejected').sum()
            st.metric("Predicted Rejections", f"{n_rej:,}")
        with col3:
            st.metric("Predicted Approvals", f"{len(hist_df) - n_rej:,}")

        st.download_button(
            "⬇️ Download Full History (CSV)",
            hist_df.to_csv(index=False).encode('utf-8'),
            "prediction_history.csv", "text/csv"
        )

        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# ============================================================================
# PAGE 8: DOCUMENTATION
# ============================================================================
else:
    st.markdown("### 📚 Documentation")
    st.markdown("---")

    st.markdown("""
## 📖 Loan Rejection Prediction System

### Overview
This application uses machine learning to predict loan rejection risk based on
financial data from the Survey of Consumer Finances (SCF) 2019 dataset.

---

### 🆕 Key Features
- **Robust loading**: the app checks several likely locations for the dataset and
  model, and lets you upload either directly from the browser if they're missing.
- **Batch Prediction**: score many applicants at once from an uploaded CSV.
- **Prediction History**: every single prediction made in your session is logged
  and downloadable.
- **Live-computed fallbacks**: the Model Performance page computes a confusion
  matrix and ROC curve on the fly if pre-rendered images aren't found.
- **Defensive column handling**: pages no longer crash if a column is missing.

---

### 📊 Dataset
- **Source**: Survey of Consumer Finances (SCF) 2019
- **Records**: 23,000+ households
- **Features**: 200+ financial and demographic variables

---

### 🎯 Methodology

#### Target Variable Creation
`LOAN_REJECTED` is derived from multiple indicators:
- Credit application turned down (`TURNDOWN`)
- Fear of denial (`FEARDENIAL`)
- Bankruptcy in last 5 years (`BNKRUPLAST5`)
- Foreclosure in last 5 years (`FORECLLAST5`)
- 60+ days late on payments (`LATE60`)
- Payday loan usage (`HPAYDAY`)

#### Risk Indicators
- **High DTI**: Debt-to-Income ratio > 0.4
- **High Leverage**: Leverage ratio > 0.5
- **Low Savings**: No savings account
- **No Emergency Savings**: No emergency fund

---

### 🤖 Models Used
1. **Logistic Regression** — baseline model
2. **Random Forest** — ensemble learning, feature importance
3. **Gradient Boosting** — advanced ensemble

---

### 📁 Expected Project Structure
```
project/
├── app.py
├── SCFP2019.csv                (or data/SCFP2019.csv, dataset/SCFP2019.csv)
├── model/
│   ├── loan_rejection_model.pkl
│   ├── feature_importance.csv
│   └── visualizations/
│       ├── confusion_matrix.png
│       ├── model_comparison.png
│       └── roc_curves.png
```

If any of these files are missing, the app will still run — it will prompt you
to upload the dataset/model manually, or compute charts (confusion matrix,
ROC curve) live from the data and model instead of the pre-rendered images.
""")

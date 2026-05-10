# ============================================================
# DASHBOARD NLP — KLASIFIKASI PERTANYAAN PETANI (KISAN KCC)
# Decision Tree + BoW / N-Gram / TF-IDF
# ============================================================
# Cara menjalankan:
#   pip install streamlit pandas scikit-learn nltk plotly joblib
#   streamlit run dashboard_nlp.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import os
import warnings
import joblib
import pickle

warnings.filterwarnings('ignore')

# --- Visualisasi ---
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- NLP ---
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dashboard NLP — Klasifikasi Pertanyaan Petani",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════
# CSS KUSTOM — TEMA HIJAU PERTANIAN MODERN
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --green-900: #1B5E20; --green-800: #2E7D32; --green-700: #388E3C;
    --green-600: #43A047; --green-400: #66BB6A; --green-100: #E8F5E9;
    --green-50: #F1F8E9; --gold: #F9A825; --white: #FFFFFF;
    --gray-900: #1a1a1a; --gray-700: #374151; --gray-500: #6B7280;
    --gray-100: #F3F4F6;
    --shadow: 0 4px 24px rgba(46,125,50,0.10);
    --shadow-lg: 0 12px 48px rgba(27,94,32,0.15);
    --glass: rgba(255,255,255,0.7);
    --glass-border: rgba(255,255,255,0.3);
    --radius: 16px;
}

@keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes pulse-glow { 0%,100% { box-shadow: 0 0 20px rgba(46,125,50,0.15); } 50% { box-shadow: 0 0 35px rgba(46,125,50,0.3); } }

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }

.stApp { background: linear-gradient(160deg, #F1F8E9 0%, #FAFAFA 40%, #E8F5E9 70%, #F1F8E9 100%); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D3B0F 0%, #1B5E20 40%, #2E7D32 100%) !important;
    border-right: none; box-shadow: 4px 0 30px rgba(0,0,0,0.15);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.12) !important; border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important; border-radius: 12px; backdrop-filter: blur(10px);
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebar"] .stFileUploader > div { border-color: rgba(255,255,255,0.2) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--glass); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1.5px solid rgba(200,230,201,0.6); border-radius: var(--radius);
    padding: 18px 22px; box-shadow: var(--shadow);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    animation: fadeInUp 0.5s ease-out; overflow: visible;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-4px) scale(1.02); box-shadow: var(--shadow-lg);
    border-color: var(--green-400);
}
[data-testid="metric-container"] label {
    color: var(--green-700) !important; font-weight: 700 !important;
    font-size: 0.72rem !important; letter-spacing: 0.06em !important; text-transform: uppercase;
    white-space: normal !important; overflow: visible !important; text-overflow: unset !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--green-900) !important; font-weight: 800 !important; font-size: 1.3rem !important;
    white-space: normal !important; overflow: visible !important; text-overflow: unset !important;
    word-break: break-word !important; line-height: 1.3 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
    white-space: normal !important; overflow: visible !important; text-overflow: unset !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1B5E20, #2E7D32, #43A047) !important;
    background-size: 200% 200% !important; animation: shimmer 3s ease infinite;
    color: white !important; font-weight: 700 !important; font-size: 0.95rem !important;
    border: none !important; border-radius: 12px !important; padding: 0.65rem 2rem !important;
    letter-spacing: 0.03em; box-shadow: 0 4px 18px rgba(46,125,50,0.35) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(46,125,50,0.45) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    border: 2px solid #A5D6A7 !important; border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 0.95rem !important;
    transition: all 0.3s ease !important; background: rgba(255,255,255,0.8) !important;
}
.stTextArea textarea:focus {
    border-color: #2E7D32 !important; box-shadow: 0 0 0 4px rgba(46,125,50,0.12) !important;
    background: white !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div { border: 2px solid #A5D6A7 !important; border-radius: 12px !important; }

/* ── Alerts ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--glass); backdrop-filter: blur(10px); border-radius: var(--radius);
    padding: 6px; box-shadow: var(--shadow); gap: 6px;
    border: 1px solid rgba(200,230,201,0.4);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important; font-weight: 600 !important;
    color: var(--green-700) !important; padding: 10px 22px !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1B5E20, #2E7D32) !important; color: white !important;
    box-shadow: 0 4px 15px rgba(27,94,32,0.3) !important;
}

/* ── DataFrame ── */
.stDataFrame {
    border-radius: var(--radius) !important; box-shadow: var(--shadow) !important;
    border: 1.5px solid #C8E6C9 !important; overflow: hidden;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--glass) !important; border-radius: var(--radius) !important;
    border: 1.5px solid #C8E6C9 !important; font-weight: 600 !important; color: var(--green-800) !important;
}

/* ── Custom cards ── */
.nlp-card {
    background: var(--glass); backdrop-filter: blur(10px); border-radius: var(--radius);
    padding: 26px; border: 1.5px solid rgba(200,230,201,0.5);
    box-shadow: var(--shadow); margin-bottom: 16px;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.nlp-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-2px); }
.nlp-card h4 { color: var(--green-800); margin-bottom: 12px; font-weight: 700; }

/* ── Prediction results ── */
.pred-result-agri {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9); border: 2.5px solid #2E7D32;
    border-radius: var(--radius); padding: 30px; text-align: center;
    box-shadow: 0 8px 32px rgba(46,125,50,0.20); animation: fadeInUp 0.6s ease-out, pulse-glow 3s ease-in-out infinite;
}
.pred-result-horti {
    background: linear-gradient(135deg, #F1F8E9, #DCEDC8); border: 2.5px solid #558B2F;
    border-radius: var(--radius); padding: 30px; text-align: center;
    box-shadow: 0 8px 32px rgba(85,139,47,0.20); animation: fadeInUp 0.6s ease-out, pulse-glow 3s ease-in-out infinite;
}
.pred-label { font-size: 2.4rem; font-weight: 800; margin: 8px 0; }
.pred-conf { font-size: 1rem; color: var(--gray-700); font-weight: 500; }

/* ── Step cards ── */
.step-card {
    background: var(--glass); backdrop-filter: blur(8px);
    border-left: 5px solid #2E7D32; border-radius: 0 var(--radius) var(--radius) 0;
    padding: 18px 22px; margin: 10px 0; box-shadow: var(--shadow);
    transition: all 0.3s ease; animation: fadeInUp 0.5s ease-out;
}
.step-card:hover { transform: translateX(6px); box-shadow: var(--shadow-lg); }
.step-num { font-size: 0.72rem; font-weight: 700; color: #2E7D32; text-transform: uppercase; letter-spacing: 0.1em; }
.step-title { font-size: 1.05rem; font-weight: 700; color: #1B5E20; margin: 4px 0; }
.step-desc { font-size: 0.85rem; color: #555; line-height: 1.5; }

/* ── Code ── */
code {
    background: #E8F5E9 !important; color: #1B5E20 !important;
    border-radius: 6px !important; padding: 2px 8px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
}

hr { border-color: #C8E6C9 !important; }
.stProgress > div > div { background: linear-gradient(90deg, #1B5E20, #43A047, #66BB6A) !important; border-radius: 100px !important; }
.stSpinner > div { border-top-color: #2E7D32 !important; }

/* ── Badges ── */
.badge-agri { background: linear-gradient(135deg,#1B5E20,#2E7D32); color: white; padding: 5px 14px; border-radius: 100px; font-size: 0.78rem; font-weight: 700; display: inline-block; }
.badge-horti { background: linear-gradient(135deg,#558B2F,#689F38); color: white; padding: 5px 14px; border-radius: 100px; font-size: 0.78rem; font-weight: 700; display: inline-block; }
.badge-best { background: linear-gradient(135deg,#F9A825,#FFC107); color: #1a1a1a; padding: 5px 14px; border-radius: 100px; font-size: 0.78rem; font-weight: 700; display: inline-block; }

/* ── Footer ── */
.dashboard-footer {
    text-align: center; padding: 30px 20px; margin-top: 60px;
    border-top: 1.5px solid #C8E6C9; color: #888; font-size: 0.82rem;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 100px; font-size: 0.75rem; font-weight: 600;
}
.status-loaded { background: rgba(76,175,80,0.2); color: #A5D6A7; }
.status-empty { background: rgba(255,193,7,0.2); color: #FFD54F; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# KONSTANTA & WARNA
# ════════════════════════════════════════════════════════════════
COLOR_BOW    = "#1B5E20"
COLOR_NGRAM  = "#388E3C"
COLOR_TFIDF  = "#66BB6A"
COLOR_GOLD   = "#F9A825"
COLORS_3     = [COLOR_BOW, COLOR_NGRAM, COLOR_TFIDF]
SCENARIO_NAMES = ["DT + BoW", "DT + N-Gram", "DT + TF-IDF"]

N_ROWS = 10_000
TARGET_CLASSES = ["AGRICULTURE", "HORTICULTURE"]
DATASET_FILENAME = "query_agg.csv"


# ════════════════════════════════════════════════════════════════
# FUNGSI PREPROCESSING
# ════════════════════════════════════════════════════════════════
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def preprocessing_steps(text: str):
    step0 = str(text)
    step1 = step0.lower()
    step2 = re.sub(r"[^a-z\s]", " ", step1)
    step3 = re.sub(r"\s+", " ", step2).strip()
    tokens = step3.split()
    step4 = " ".join([t for t in tokens if t not in STOP_WORDS and len(t) > 1])
    return {"original": step0, "lowercase": step1,
            "no_punct": step2, "normalized": step3, "clean": step4}


# ════════════════════════════════════════════════════════════════
# LOAD & CACHE DATA
# ════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(path: str):
    df = pd.read_csv(path, nrows=N_ROWS, encoding="utf-8", on_bad_lines="skip")
    return df

@st.cache_resource(show_spinner=False)
def prepare_data(_df: pd.DataFrame):
    df2 = _df[["QueryText", "Sector"]].dropna()
    df2 = df2[df2["Sector"].isin(TARGET_CLASSES)].copy()
    df2["Sector"] = df2["Sector"].str.strip().str.upper()
    df2["clean_text"] = df2["QueryText"].apply(clean_text)
    le = LabelEncoder()
    df2["label"] = le.fit_transform(df2["Sector"])
    return df2, le

@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file):
    """Load data from uploaded file (separate cache from local file)."""
    df = pd.read_csv(uploaded_file, nrows=N_ROWS, encoding="utf-8", on_bad_lines="skip")
    return df


# ════════════════════════════════════════════════════════════════
# TRAINING & EVALUASI
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_all_models(_df: pd.DataFrame, _le):
    X = _df["clean_text"]
    y = _df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Vectorizers
    vec_bow   = CountVectorizer(ngram_range=(1,1), max_features=5000, min_df=2)
    vec_ngram = CountVectorizer(ngram_range=(2,3), max_features=5000, min_df=2)
    vec_tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True, min_df=2)

    Xtr_bow   = vec_bow.fit_transform(X_train);    Xte_bow   = vec_bow.transform(X_test)
    Xtr_ngram = vec_ngram.fit_transform(X_train);  Xte_ngram = vec_ngram.transform(X_test)
    Xtr_tfidf = vec_tfidf.fit_transform(X_train);  Xte_tfidf = vec_tfidf.transform(X_test)

    DT_PARAMS = dict(criterion="gini", max_depth=20,
                     min_samples_split=5, min_samples_leaf=2, random_state=42)

    scenarios = [
        ("DT + BoW",    Xtr_bow,   Xte_bow,   vec_bow),
        ("DT + N-Gram", Xtr_ngram, Xte_ngram, vec_ngram),
        ("DT + TF-IDF", Xtr_tfidf, Xte_tfidf, vec_tfidf),
    ]

    models, results, predictions, vectorizers, train_times = {}, {}, {}, {}, {}

    for name, Xtr, Xte, vec in scenarios:
        t0 = time.time()
        m  = DecisionTreeClassifier(**DT_PARAMS)
        m.fit(Xtr, y_train)
        elapsed = time.time() - t0

        y_pred = m.predict(Xte)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        cr   = classification_report(y_test, y_pred,
                                      target_names=_le.classes_, zero_division=0)

        models[name]      = m
        vectorizers[name] = vec
        train_times[name] = elapsed
        predictions[name] = y_pred
        results[name]     = {"accuracy": acc, "precision": prec, "recall": rec,
                              "f1_score": f1, "cm": cm, "report": cr,
                              "depth": m.get_depth(), "leaves": m.get_n_leaves()}

    # Skenario terbaik
    best_name = max(results, key=lambda x: results[x]["accuracy"])
    best_model   = models[best_name]
    best_vec     = vectorizers[best_name]

    # Simpan model terbaik ke file
    try:
        joblib.dump(best_model, "best_model.pkl")
        joblib.dump(best_vec,   "best_vectorizer.pkl")
        joblib.dump(_le,         "label_encoder.pkl")
    except Exception:
        pass

    return (models, vectorizers, results, predictions, train_times,
            y_test, _le, best_name, best_model, best_vec,
            X_train, X_test, y_train)


# ════════════════════════════════════════════════════════════════
# HELPER PLOTLY
# ════════════════════════════════════════════════════════════════
def plotly_bar(names, values, title, ylabel, colors=COLORS_3):
    fig = go.Figure()
    for i, (n, v) in enumerate(zip(names, values)):
        fig.add_trace(go.Bar(
            x=[n], y=[v], name=n,
            marker_color=colors[i],
            text=[f"{v:.4f}"], textposition="outside",
            width=0.5
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Plus Jakarta Sans", color="#1B5E20")),
        yaxis_title=ylabel, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#E8F5E9", tickfont=dict(family="Plus Jakarta Sans")),
        xaxis=dict(tickfont=dict(family="Plus Jakarta Sans")),
        margin=dict(t=50, b=20, l=20, r=20),
        font=dict(family="Plus Jakarta Sans")
    )
    return fig

def plotly_cm(cm, title, labels, color):
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                    x=labels, y=labels,
                    color_continuous_scale=[[0,"#FFFFFF"],[1,color]],
                    aspect="auto")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, family="Plus Jakarta Sans", color="#1B5E20")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Plus Jakarta Sans"),
        coloraxis_showscale=False,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    fig.update_traces(textfont=dict(size=16, color="white"))
    return fig

def plotly_radar(results_dict):
    cats = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig  = go.Figure()
    for i, (name, r) in enumerate(results_dict.items()):
        vals = [r["accuracy"], r["precision"], r["recall"], r["f1_score"]]
        vals_c = vals + [vals[0]]
        cats_c = cats + [cats[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats_c, fill="toself",
            name=name, line=dict(color=COLORS_3[i], width=2.5),
            fillcolor=COLORS_3[i], opacity=0.25
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                           gridcolor="#C8E6C9", tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11, family="Plus Jakarta Sans"))
        ),
        showlegend=True,
        legend=dict(font=dict(family="Plus Jakarta Sans", size=11)),
        paper_bgcolor="white",
        margin=dict(t=40, b=40, l=60, r=60),
        font=dict(family="Plus Jakarta Sans")
    )
    return fig


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 24px 0 12px 0;'>
            <div style='font-size:3.2rem; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.2));'>🌾</div>
            <div style='font-size:1.15rem; font-weight:800; letter-spacing:0.04em;
                        margin-top:10px; text-shadow: 0 1px 3px rgba(0,0,0,0.2);'>NLP Dashboard</div>
            <div style='font-size:0.75rem; opacity:0.8; margin-top:4px; letter-spacing:0.06em;
                        text-transform:uppercase;'>
                Kisan Query Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        menu = st.selectbox("🗂️ Navigasi Halaman", [
            "🏠  Beranda & Info Proyek",
            "📂  Dataset Overview",
            "🧹  Pre-processing Teks",
            "⚙️  Feature Extraction",
            "🤖  Training & Evaluasi",
            "📊  Perbandingan Skenario",
            "🎯  Prediksi Interaktif",
        ], label_visibility="visible")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Auto-detect local dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_csv = os.path.join(script_dir, DATASET_FILENAME)
        has_local = os.path.isfile(local_csv)

        if has_local:
            st.markdown(f"""
            <div style='background:rgba(76,175,80,0.15); border:1px solid rgba(76,175,80,0.3);
                        border-radius:12px; padding:12px 16px; margin-bottom:12px;'>
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                    <span style='font-size:1.1rem;'>✅</span>
                    <span style='font-weight:700; font-size:0.82rem;'>Dataset Terdeteksi</span>
                </div>
                <div style='font-size:0.72rem; opacity:0.85; line-height:1.5;'>
                    File <code style='background:rgba(255,255,255,0.15); padding:1px 6px;
                    border-radius:4px;'>{DATASET_FILENAME}</code> ditemukan di folder lokal.
                    Data akan di-load otomatis.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("**📁 Upload Dataset**")

        uploaded = st.file_uploader(DATASET_FILENAME, type=["csv"],
                                     help="Upload file query_agg.csv dari Kaggle (opsional jika file sudah ada di folder)",
                                     label_visibility="collapsed" if has_local else "visible")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Info proyek
        st.markdown("""
        <div style='font-size:0.78rem; opacity:0.9; line-height:1.8;'>
            <div style='font-weight:700; margin-bottom:8px; font-size:0.82rem;
                        letter-spacing:0.04em;'>📌 Info Proyek</div>
            <div>🎯 Task: Text Classification</div>
            <div>🌿 Label: Agriculture / Horticulture</div>
            <div>🌳 Model: Decision Tree</div>
            <div>📝 Fitur: BoW · N-Gram · TF-IDF</div>
            <div>📊 Data: KCC India 2013–2021</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; font-size:0.68rem; opacity:0.5; padding:10px 0;'>
            v2.0 · Streamlit Dashboard<br>Mini-Project NLP 2024
        </div>
        """, unsafe_allow_html=True)

    return menu, uploaded, has_local, local_csv


# ════════════════════════════════════════════════════════════════
# HALAMAN 1: BERANDA
# ════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #43A047 100%);
                border-radius: 20px; padding: 48px 40px; margin-bottom: 32px;
                box-shadow: 0 12px 48px rgba(27,94,32,0.3);'>
        <div style='font-size:0.85rem; color:#A5D6A7; font-weight:600;
                    letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px;'>
            Mini-Project Natural Language Processing
        </div>
        <div style='font-size:2.4rem; font-weight:800; color:white; line-height:1.2;
                    margin-bottom:16px;'>
            Klasifikasi Teks<br>Pertanyaan Petani 🌾
        </div>
        <div style='font-size:1rem; color:#C8E6C9; max-width: 600px; line-height:1.6;'>
            Sistem cerdas untuk mengklasifikasikan pertanyaan petani India ke sektor
            <strong style='color:white;'>Agriculture</strong> atau
            <strong style='color:white;'>Horticulture</strong> menggunakan
            Decision Tree dengan tiga pendekatan representasi fitur teks.
        </div>
        <div style='margin-top: 24px; display:flex; gap:12px; flex-wrap:wrap;'>
            <span style='background:rgba(255,255,255,0.2); color:white; padding:6px 16px;
                         border-radius:100px; font-size:0.82rem; font-weight:600;'>
                🌳 Decision Tree
            </span>
            <span style='background:rgba(255,255,255,0.2); color:white; padding:6px 16px;
                         border-radius:100px; font-size:0.82rem; font-weight:600;'>
                📝 Bag of Words
            </span>
            <span style='background:rgba(255,255,255,0.2); color:white; padding:6px 16px;
                         border-radius:100px; font-size:0.82rem; font-weight:600;'>
                🔤 N-Gram
            </span>
            <span style='background:rgba(255,255,255,0.2); color:white; padding:6px 16px;
                         border-radius:100px; font-size:0.82rem; font-weight:600;'>
                📊 TF-IDF
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Info cards — custom HTML agar teks tidak terpotong
    info_cards = [
        ("🗄️", "Dataset", "Kisan Query", "KCC India 2013–2021"),
        ("🌱", "Kelas Target", "2 Kelas", "Agriculture · Horticulture"),
        ("🌳", "Algoritma", "Decision Tree", "gini · depth=20"),
        ("🔬", "Skenario", "3 Eksperimen", "BoW · N-Gram · TF-IDF"),
    ]
    cols_info = st.columns(4)
    for col_i, (emoji, label, value, sub) in zip(cols_info, info_cards):
        col_i.markdown(f"""
        <div style='background:rgba(255,255,255,0.7); backdrop-filter:blur(10px);
                    border:1.5px solid rgba(200,230,201,0.6); border-radius:16px;
                    padding:20px 18px; text-align:center;
                    box-shadow:0 4px 24px rgba(46,125,50,0.10);
                    transition:all 0.3s ease;'>
            <div style='font-size:1.8rem; margin-bottom:6px;'>{emoji}</div>
            <div style='font-size:0.7rem; font-weight:700; color:#388E3C;
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>{label}</div>
            <div style='font-size:1.3rem; font-weight:800; color:#1B5E20;
                        line-height:1.2; margin-bottom:6px;'>{value}</div>
            <div style='font-size:0.72rem; color:#43A047; font-weight:600;'>↑ {sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline
    st.markdown("### 🗺️ Alur Pipeline Proyek")
    steps = [
        ("1","📂 Load Dataset","Membaca 10.000 baris dari query_agg.csv (996 MB)"),
        ("2","🧹 Pre-processing","Lowercase → Punctuation Removal → Stopword Removal"),
        ("3","✂️ Split Data","80% Training / 20% Testing (Stratified)"),
        ("4","⚙️ Feature Extraction","3 metode: BoW, N-Gram, TF-IDF"),
        ("5","🌳 Training Model","Decision Tree untuk 3 skenario"),
        ("6","📊 Evaluasi","Accuracy, Precision, Recall, F1-Score"),
        ("7","💾 Simpan Model","joblib.dump() → .pkl file"),
        ("8","🎯 Prediksi","Real-time prediction via dashboard"),
    ]
    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-num'>Langkah {num}</div>
                <div class='step-title'>{title}</div>
                <div class='step-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Info dataset
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='nlp-card'>
            <h4>📌 Tentang Dataset</h4>
            <p style='color:#555; font-size:0.92rem; line-height:1.7;'>
                <strong>Kisan Query Analysis Dataset</strong> berasal dari program
                <em>Kisan Call Centre (KCC)</em> pemerintah India — layanan konsultasi
                pertanian via telepon yang aktif sejak 2004. Dataset ini merekam jutaan
                pertanyaan petani dari seluruh negara bagian India beserta jawabannya,
                periode 2013–2021.
            </p>
            <p style='color:#555; font-size:0.92rem;'>
                📥 Sumber:
                <a href='https://www.kaggle.com/datasets/anirudhvadakedath/kisan-query-analysis-dataset'
                   target='_blank' style='color:#2E7D32; font-weight:600;'>Kaggle</a>
                (996.14 MB, file: query_agg.csv)
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='nlp-card'>
            <h4>🎯 Tujuan Bisnis</h4>
            <p style='color:#555; font-size:0.92rem; line-height:1.7;'>
                Membangun sistem klasifikasi otomatis untuk merutekan pertanyaan petani
                ke ahli yang tepat — ahli <strong>agronomi</strong> (Agriculture) atau
                ahli <strong>hortikultura</strong> (Horticulture) — berdasarkan teks
                pertanyaan, tanpa intervensi manual.
            </p>
            <ul style='color:#555; font-size:0.88rem; line-height:1.9; padding-left:20px;'>
                <li>⚡ Mempercepat waktu respons KCC</li>
                <li>✅ Mengurangi human error routing</li>
                <li>📈 Meningkatkan akurasi layanan pertanian</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HALAMAN 2: DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════
def page_dataset(df_raw, df_clean):
    st.markdown("## 📂 Dataset Overview")
    st.markdown("Eksplorasi data mentah dari Kisan Query Analysis Dataset.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Metrik dasar
    total     = len(df_raw)
    n_clean   = len(df_clean)
    n_agri    = (df_clean["Sector"] == "AGRICULTURE").sum()
    n_horti   = (df_clean["Sector"] == "HORTICULTURE").sum()
    n_miss    = df_raw[["QueryText","Sector"]].isnull().sum().sum()
    n_cols    = df_raw.shape[1]

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("📋 Total Baris", f"{total:,}", "setelah nrows=10k")
    c2.metric("✅ Data Bersih", f"{n_clean:,}", "setelah filter")
    c3.metric("🌾 Agriculture", f"{n_agri:,}", f"{n_agri/n_clean*100:.1f}%")
    c4.metric("🌿 Horticulture", f"{n_horti:,}", f"{n_horti/n_clean*100:.1f}%")
    c5.metric("❓ Missing Values", f"{n_miss:,}", "QueryText+Sector")
    c6.metric("📊 Jumlah Kolom", f"{n_cols}", "kolom asli")

    st.markdown("<br>", unsafe_allow_html=True)

    # Grafik distribusi
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📊 Distribusi Kelas (Sector)")
        vc  = df_clean["Sector"].value_counts()
        fig = go.Figure(go.Bar(
            x=vc.index, y=vc.values,
            marker_color=[COLOR_BOW, COLOR_NGRAM],
            text=vc.values, textposition="outside",
            width=0.5
        ))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#E8F5E9"),
            margin=dict(t=20,b=20,l=20,r=20),
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 🥧 Proporsi Kelas")
        fig2 = go.Figure(go.Pie(
            labels=vc.index, values=vc.values,
            hole=0.5,
            marker_colors=[COLOR_BOW, COLOR_NGRAM],
            textfont=dict(size=13, family="Plus Jakarta Sans"),
            pull=[0.05, 0.05]
        ))
        fig2.update_layout(
            paper_bgcolor="white",
            legend=dict(font=dict(family="Plus Jakarta Sans")),
            margin=dict(t=20,b=20,l=20,r=20),
            annotations=[dict(text=f"{n_clean:,}<br>Data", x=0.5, y=0.5,
                              font_size=14, showarrow=False,
                              font=dict(family="Plus Jakarta Sans", color="#1B5E20"))]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Distribusi panjang teks
    st.markdown("#### 📏 Distribusi Panjang Teks QueryText")
    df_viz = df_clean.copy()
    df_viz["text_len"]  = df_viz["QueryText"].str.len()
    df_viz["word_count"] = df_viz["QueryText"].str.split().str.len()

    col_p, col_q = st.columns(2)
    with col_p:
        fig3 = px.histogram(df_viz, x="text_len", nbins=60,
                             title="Jumlah Karakter",
                             color_discrete_sequence=[COLOR_BOW])
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           font=dict(family="Plus Jakarta Sans"),
                           margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig3, use_container_width=True)
    with col_q:
        fig4 = px.histogram(df_viz, x="word_count", nbins=50,
                             title="Jumlah Kata",
                             color_discrete_sequence=[COLOR_NGRAM])
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           font=dict(family="Plus Jakarta Sans"),
                           margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig4, use_container_width=True)

    # Preview data
    st.markdown("#### 👁️ Preview Dataset (10 Baris)")
    cols_show = [c for c in ["StateName","Season","Sector","Category",
                              "Crop","QueryText"] if c in df_raw.columns]
    st.dataframe(df_raw[cols_show].head(10), use_container_width=True, height=300)

    # Statistik
    with st.expander("📈 Statistik Deskriptif Kolom Teks"):
        st.dataframe(df_raw[["QueryText"]].describe() if "QueryText" in df_raw.columns
                     else df_raw.describe(), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# HALAMAN 3: PRE-PROCESSING
# ════════════════════════════════════════════════════════════════
def page_preprocessing(df_clean):
    st.markdown("## 🧹 Pre-processing Teks")
    st.markdown("Pipeline pembersihan teks sebelum transformasi fitur.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Langkah-langkah
    st.markdown("### 📋 Pipeline Pre-processing")
    steps_info = [
        ("1", "Data Reduction",     "Ambil 10.000 baris pertama dari file 996 MB untuk efisiensi Google Colab.", "nrows=10000"),
        ("2", "Filter Label",       "Buang kelas 'Other'. Pertahankan hanya AGRICULTURE dan HORTICULTURE.", "isin(['AGRICULTURE','HORTICULTURE'])"),
        ("3", "Drop Missing Values","Buang baris dengan nilai kosong pada QueryText dan Sector.", "dropna()"),
        ("4", "Case Folding",       "Ubah semua teks menjadi huruf kecil (lowercase).", ".lower()"),
        ("5", "Punctuation Removal","Hapus tanda baca dan karakter non-huruf menggunakan regex.", "re.sub(r'[^a-z\\s]')"),
        ("6", "Stopword Removal",   "Hapus kata-kata umum bahasa Inggris yang tidak informatif.", "NLTK English stopwords"),
        ("7", "Label Encoding",     "Ubah label AGRICULTURE→0 dan HORTICULTURE→1.", "LabelEncoder()"),
    ]

    for s in steps_info:
        st.markdown(f"""
        <div class='step-card'>
            <div class='step-num'>Langkah {s[0]}</div>
            <div class='step-title'>{s[1]}</div>
            <div class='step-desc'>{s[2]}</div>
            <code style='margin-top:6px; display:inline-block;'>{s[3]}</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Demo interaktif
    st.markdown("### 🔬 Demo Preprocessing Interaktif")
    st.markdown("Masukkan teks untuk melihat hasil setiap tahap preprocessing:")

    col_demo, _ = st.columns([2, 1])
    with col_demo:
        demo_text = st.text_area("Teks input:", height=100,
            value="My paddy leaves are turning YELLOW! I see small insects on the stem?? What should I do?",
            key="preproc_demo")

    if demo_text.strip():
        steps_result = preprocessing_steps(demo_text)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔄 Hasil Setiap Tahap:")

        step_labels = [
            ("📄 Teks Asli",           "original"),
            ("🔡 Setelah Lowercase",    "lowercase"),
            ("✂️ Hapus Tanda Baca",     "no_punct"),
            ("📏 Normalisasi Spasi",    "normalized"),
            ("🧹 Setelah Stopwords",    "clean"),
        ]
        for label, key in step_labels:
            val = steps_result[key]
            color = "#E8F5E9" if key == "clean" else "#F9FBE7"
            border = "#2E7D32" if key == "clean" else "#A5D6A7"
            st.markdown(f"""
            <div style='background:{color}; border:1.5px solid {border};
                        border-radius:10px; padding:14px 18px; margin:6px 0;'>
                <div style='font-size:0.78rem; font-weight:700; color:#2E7D32;
                            text-transform:uppercase; letter-spacing:0.06em;
                            margin-bottom:6px;'>{label}</div>
                <div style='font-family:"JetBrains Mono",monospace; font-size:0.9rem;
                            color:#1B5E20; word-break:break-all;'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # Perbandingan before vs after
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Perbandingan Before vs After (10 Sampel)")
    sample = df_clean[["QueryText","clean_text","Sector"]].head(10)
    sample.columns = ["QueryText (Sebelum)", "clean_text (Sesudah)", "Sektor"]
    st.dataframe(sample, use_container_width=True, height=320)


# ════════════════════════════════════════════════════════════════
# HALAMAN 4: FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════
def page_feature_extraction(vectorizers=None, df_clean=None):
    st.markdown("## ⚙️ Feature Extraction")
    st.markdown("Tiga metode transformasi teks menjadi representasi numerik.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Show actual vectorizer stats if available
    if vectorizers is not None and df_clean is not None:
        st.markdown("### 📊 Statistik Fitur dari Data Aktual")
        c1, c2, c3 = st.columns(3)
        vec_info = [
            ("DT + BoW", "📝 Bag of Words", COLOR_BOW),
            ("DT + N-Gram", "🔤 N-Gram (2,3)", COLOR_NGRAM),
            ("DT + TF-IDF", "📊 TF-IDF", COLOR_TFIDF),
        ]
        for col, (vname, vlabel, vcol) in zip([c1,c2,c3], vec_info):
            if vname in vectorizers:
                v = vectorizers[vname]
                n_feat = len(v.get_feature_names_out())
                col.metric(vlabel, f"{n_feat:,} fitur", f"max_features=5000")

        # Top words bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔝 Top 15 Fitur per Metode")
        tabs_feat = st.tabs(["📝 BoW Top Words", "🔤 N-Gram Top Phrases", "📊 TF-IDF Top Terms"])
        for tab_f, (vname, vlabel, vcol) in zip(tabs_feat, vec_info):
            with tab_f:
                if vname in vectorizers:
                    vec = vectorizers[vname]
                    feature_names = vec.get_feature_names_out()
                    X_vec = vec.transform(df_clean["clean_text"])
                    sums = X_vec.sum(axis=0).A1
                    top_idx = sums.argsort()[-15:][::-1]
                    top_words = [feature_names[i] for i in top_idx]
                    top_vals = [sums[i] for i in top_idx]

                    fig = go.Figure(go.Bar(
                        x=top_vals[::-1], y=top_words[::-1],
                        orientation='h', marker_color=vcol,
                        text=[f"{tv:,.0f}" for tv in top_vals[::-1]],
                        textposition="outside"
                    ))
                    fig.update_layout(
                        title=dict(text=f"Top 15 Fitur — {vlabel}",
                                   font=dict(size=14, family="Plus Jakarta Sans", color="#1B5E20")),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(gridcolor="#E8F5E9", title="Frekuensi / Skor"),
                        margin=dict(t=50, b=20, l=120, r=40),
                        height=420, font=dict(family="Plus Jakarta Sans")
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📖 Penjelasan Metode")
    tab1, tab2, tab3 = st.tabs(["📝 Bag of Words (BoW)", "🔤 N-Gram", "📊 TF-IDF"])

    with tab1:
        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.markdown("""
            <div class='nlp-card'>
                <h4 style='color:#1B5E20;'>📝 Bag of Words (BoW)</h4>
                <p style='color:#555; font-size:0.92rem; line-height:1.7;'>
                    Metode paling sederhana yang menghitung frekuensi kemunculan setiap kata
                    dalam dokumen tanpa memperhatikan urutan. Setiap dokumen direpresentasikan
                    sebagai vektor frekuensi kata dari seluruh kosakata.
                </p>
                <hr style='border-color:#C8E6C9;'>
                <div style='font-weight:700; color:#2E7D32; margin-bottom:8px;'>⚙️ Parameter:</div>
                <code>CountVectorizer(ngram_range=(1,1), max_features=5000, min_df=2)</code>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                <div style='background:#E8F5E9; border-radius:10px; padding:14px;'>
                    <div style='font-weight:700; color:#1B5E20; font-size:0.85rem;'>✅ Kelebihan</div>
                    <ul style='color:#555; font-size:0.83rem; margin:8px 0; padding-left:16px; line-height:1.8;'>
                        <li>Sederhana & mudah dipahami</li>
                        <li>Komputasi sangat cepat</li>
                        <li>Efektif sebagai baseline</li>
                        <li>Cocok untuk Decision Tree</li>
                    </ul>
                </div>
                <div style='background:#FFF3E0; border-radius:10px; padding:14px;'>
                    <div style='font-weight:700; color:#E65100; font-size:0.85rem;'>❌ Kekurangan</div>
                    <ul style='color:#555; font-size:0.83rem; margin:8px 0; padding-left:16px; line-height:1.8;'>
                        <li>Mengabaikan urutan kata</li>
                        <li>Tidak menangkap frasa</li>
                        <li>Kata umum diberi bobot sama</li>
                        <li>Tidak ada makna semantik</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_r:
            st.markdown("""
            <div class='nlp-card'>
                <h4 style='color:#1B5E20;'>💡 Ilustrasi</h4>
                <p style='color:#555; font-size:0.88rem;'><strong>Input:</strong><br>
                "wheat pest control wheat"</p>
                <div style='background:#F1F8E9; border-radius:8px; padding:12px;
                            font-family:"JetBrains Mono",monospace; font-size:0.82rem;
                            margin:10px 0; color:#1B5E20;'>
                    Vocabulary:<br>
                    {wheat:0, pest:1, control:2}<br><br>
                    Vektor: [2, 1, 1]<br>
                    → wheat=2, pest=1, control=1
                </div>
                <p style='color:#777; font-size:0.82rem; margin-top:8px;'>
                    Urutan kata tidak diperhatikan — hanya frekuensi yang dihitung.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.markdown("""
            <div class='nlp-card'>
                <h4 style='color:#1B5E20;'>🔤 N-Gram</h4>
                <p style='color:#555; font-size:0.92rem; line-height:1.7;'>
                    Perluasan BoW yang menangkap pasangan atau kelompok kata berurutan.
                    Dalam proyek ini digunakan <strong>bigram + trigram</strong> untuk
                    menangkap frasa pertanian yang bermakna seperti "stem borer",
                    "paddy cultivation", "drip irrigation".
                </p>
                <hr style='border-color:#C8E6C9;'>
                <div style='font-weight:700; color:#2E7D32; margin-bottom:8px;'>⚙️ Parameter:</div>
                <code>CountVectorizer(ngram_range=(2,3), max_features=5000, min_df=2)</code>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                <div style='background:#E8F5E9; border-radius:10px; padding:14px;'>
                    <div style='font-weight:700; color:#1B5E20; font-size:0.85rem;'>✅ Kelebihan</div>
                    <ul style='color:#555; font-size:0.83rem; margin:8px 0; padding-left:16px; line-height:1.8;'>
                        <li>Menangkap frasa bermakna</li>
                        <li>Cocok untuk istilah teknis</li>
                        <li>Konteks lebih kaya</li>
                    </ul>
                </div>
                <div style='background:#FFF3E0; border-radius:10px; padding:14px;'>
                    <div style='font-weight:700; color:#E65100; font-size:0.85rem;'>❌ Kekurangan</div>
                    <ul style='color:#555; font-size:0.83rem; margin:8px 0; padding-left:16px; line-height:1.8;'>
                        <li>Matriks sangat sparse</li>
                        <li>Butuh lebih banyak data</li>
                        <li>Rentan overfitting</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_r:
            st.markdown("""
            <div class='nlp-card'>
                <h4 style='color:#1B5E20;'>💡 Ilustrasi</h4>
                <p style='color:#555; font-size:0.88rem;'><strong>Input:</strong><br>
                "wheat pest control"</p>
                <div style='background:#F1F8E9; border-radius:8px; padding:12px;
                            font-family:"JetBrains Mono",monospace; font-size:0.82rem;
                            margin:10px 0; color:#1B5E20;'>
                    Bigram:<br>
                    [wheat pest] [pest control]<br><br>
                    Trigram:<br>
                    [wheat pest control]
                </div>
                <p style='color:#777; font-size:0.82rem; margin-top:8px;'>
                    Frasa "stem borer" lebih diskriminatif daripada kata tunggal.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.markdown("""
            <div class='nlp-card'>
                <h4 style='color:#1B5E20;'>📊 TF-IDF</h4>
                <p style='color:#555; font-size:0.92rem; line-height:1.7;'>
                    Metode paling canggih yang memberikan bobot pada kata berdasarkan
                    frekuensi di dokumen (TF) <em>dan</em> seberapa jarang kata tersebut
                    muncul di seluruh korpus (IDF). Kata umum mendapat bobot rendah,
                    kata spesifik mendapat bobot tinggi.
                </p>
                <hr style='border-color:#C8E6C9;'>
                <div style='font-weight:700; color:#2E7D32; margin-bottom:8px;'>⚙️ Parameter:</div>
                <code>TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)</code>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                <div style='background:#E8F5E9; border-radius:10px; padding:14px;'>
                    <div style='font-weight:700; color:#1B5E20; font-size:0.85rem;'>✅ Kelebihan</div>
                    <ul style='color:#555; font-size:0.83rem; margin:8px 0; padding-left:16px; line-height:1.8;'>
                        <li>Pembobotan otomatis cerdas</li>
                        <li>Menekan kata tidak informatif</li>
                        <li>Standar industri NLP</li>
                    </ul>
                </div>
                <div style='background:#FFF3E0; border-radius:10px; padding:14px;'>
                    <div style='font-weight:700; color:#E65100; font-size:0.85rem;'>❌ Kekurangan</div>
                    <ul style='color:#555; font-size:0.83rem; margin:8px 0; padding-left:16px; line-height:1.8;'>
                        <li>Lebih kompleks dari BoW</li>
                        <li>Tidak menangkap makna</li>
                        <li>IDF hanya dari data train</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_r:
            st.markdown("""
            <div class='nlp-card'>
                <h4 style='color:#1B5E20;'>💡 Formula</h4>
                <div style='background:#F1F8E9; border-radius:8px; padding:12px;
                            font-family:"JetBrains Mono",monospace; font-size:0.8rem;
                            margin:10px 0; color:#1B5E20; line-height:1.8;'>
                    TF-IDF(t,d) = TF(t,d) × IDF(t)<br><br>
                    TF: frekuensi kata t di doc d<br>
                    IDF: log(N / df(t)) + 1<br><br>
                    paddy → IDF tinggi (spesifik) ✅<br>
                    crop  → IDF rendah (umum) ❌
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HALAMAN 5: TRAINING & EVALUASI
# ════════════════════════════════════════════════════════════════
def page_training(results, train_times, predictions, y_test, le):
    st.markdown("## 🤖 Training & Evaluasi Model")
    st.markdown("Hasil pelatihan Decision Tree pada 3 skenario representasi fitur.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs per skenario
    tab1, tab2, tab3 = st.tabs([
        "🟢 DT + BoW", "🟩 DT + N-Gram", "🌿 DT + TF-IDF"
    ])

    scenario_colors = {
        "DT + BoW": COLOR_BOW,
        "DT + N-Gram": COLOR_NGRAM,
        "DT + TF-IDF": COLOR_TFIDF
    }

    for tab, sname in zip([tab1, tab2, tab3], SCENARIO_NAMES):
        with tab:
            r = results[sname]
            col_color = scenario_colors[sname]

            # Metrik utama
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("🎯 Accuracy",  f"{r['accuracy']*100:.2f}%")
            c2.metric("📐 Precision", f"{r['precision']:.4f}")
            c3.metric("🔍 Recall",    f"{r['recall']:.4f}")
            c4.metric("⚖️ F1-Score",  f"{r['f1_score']:.4f}")
            c5.metric("⏱️ Waktu",     f"{train_times[sname]:.3f}s")

            st.markdown("<br>", unsafe_allow_html=True)

            col_cm, col_rep = st.columns([1, 1.2])
            with col_cm:
                st.markdown("#### 🗂️ Confusion Matrix")
                fig_cm = plotly_cm(
                    r["cm"], f"Confusion Matrix — {sname}",
                    list(le.classes_), col_color
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                # Info TP, TN, FP, FN
                cm = r["cm"]
                TP = int(cm[0,0]); FP = int(cm[0,1])
                FN = int(cm[1,0]); TN = int(cm[1,1])
                st.markdown(f"""
                <div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:8px;'>
                    <div style='background:#E8F5E9; border-radius:8px; padding:10px; text-align:center;'>
                        <div style='font-size:1.3rem; font-weight:800; color:#1B5E20;'>{TP}</div>
                        <div style='font-size:0.72rem; color:#555; font-weight:600;'>TRUE POS</div>
                    </div>
                    <div style='background:#FFF3E0; border-radius:8px; padding:10px; text-align:center;'>
                        <div style='font-size:1.3rem; font-weight:800; color:#E65100;'>{FP}</div>
                        <div style='font-size:0.72rem; color:#555; font-weight:600;'>FALSE POS</div>
                    </div>
                    <div style='background:#FFF3E0; border-radius:8px; padding:10px; text-align:center;'>
                        <div style='font-size:1.3rem; font-weight:800; color:#E65100;'>{FN}</div>
                        <div style='font-size:0.72rem; color:#555; font-weight:600;'>FALSE NEG</div>
                    </div>
                    <div style='background:#E8F5E9; border-radius:8px; padding:10px; text-align:center;'>
                        <div style='font-size:1.3rem; font-weight:800; color:#1B5E20;'>{TN}</div>
                        <div style='font-size:0.72rem; color:#555; font-weight:600;'>TRUE NEG</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_rep:
                st.markdown("#### 📋 Classification Report")
                st.code(r["report"], language=None)
                st.markdown(f"""
                <div class='nlp-card' style='margin-top:12px;'>
                    <h4>🌳 Info Pohon Keputusan</h4>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
                        <div style='text-align:center;'>
                            <div style='font-size:2rem; font-weight:800; color:{col_color};'>
                                {r["depth"]}
                            </div>
                            <div style='color:#555; font-size:0.82rem; font-weight:600;'>
                                Kedalaman Pohon
                            </div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-size:2rem; font-weight:800; color:{col_color};'>
                                {r["leaves"]:,}
                            </div>
                            <div style='color:#555; font-size:0.82rem; font-weight:600;'>
                                Jumlah Leaves
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HALAMAN 6: PERBANDINGAN
# ════════════════════════════════════════════════════════════════
def page_comparison(results, train_times):
    st.markdown("## 📊 Perbandingan 3 Skenario")
    st.markdown("Analisis komprehensif performa DT + BoW vs DT + N-Gram vs DT + TF-IDF.")
    st.markdown("<br>", unsafe_allow_html=True)

    names = SCENARIO_NAMES
    accs  = [results[n]["accuracy"]*100  for n in names]
    precs = [results[n]["precision"]     for n in names]
    recs  = [results[n]["recall"]        for n in names]
    f1s   = [results[n]["f1_score"]      for n in names]
    times = [train_times[n]              for n in names]

    # Summary table
    best_n = names[int(np.argmax(accs))]
    df_sum = pd.DataFrame({
        "Skenario": names,
        "Accuracy (%)": [f"{a:.2f}" for a in accs],
        "Precision": [f"{p:.4f}" for p in precs],
        "Recall": [f"{r:.4f}" for r in recs],
        "F1-Score": [f"{f:.4f}" for f in f1s],
        "Waktu (s)": [f"{t:.3f}" for t in times],
        "Status": ["🥇 TERBAIK" if n==best_n else "—" for n in names]
    })
    st.markdown("#### 📋 Tabel Ringkasan Perbandingan")
    st.dataframe(df_sum, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Grafik bar
    st.markdown("#### 📊 Grafik Perbandingan 4 Metrik")
    col1,col2 = st.columns(2)
    col3,col4 = st.columns(2)

    with col1:
        st.plotly_chart(plotly_bar(names, accs, "Accuracy (%)", "Accuracy (%)"),
                        use_container_width=True)
    with col2:
        st.plotly_chart(plotly_bar(names, precs, "Precision (Macro)", "Precision"),
                        use_container_width=True)
    with col3:
        st.plotly_chart(plotly_bar(names, recs, "Recall (Macro)", "Recall"),
                        use_container_width=True)
    with col4:
        st.plotly_chart(plotly_bar(names, f1s, "F1-Score (Macro)", "F1-Score"),
                        use_container_width=True)

    # Radar chart
    st.markdown("#### 🕸️ Radar Chart — Perbandingan Holistik")
    col_rad, col_time = st.columns([2,1])
    with col_rad:
        st.plotly_chart(plotly_radar(results), use_container_width=True)
    with col_time:
        st.plotly_chart(plotly_bar(names, times, "Waktu Training (detik)", "Detik"),
                        use_container_width=True)

    # Analisis otomatis
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Analisis Otomatis")

    best_idx   = int(np.argmax(accs))
    worst_idx  = int(np.argmin(accs))
    best_sc    = names[best_idx]
    worst_sc   = names[worst_idx]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#E8F5E9,#C8E6C9);
                    border:2px solid #2E7D32; border-radius:14px; padding:20px;'>
            <div style='font-size:0.75rem; font-weight:700; color:#2E7D32;
                        text-transform:uppercase; letter-spacing:0.08em;'>
                🥇 Skenario Terbaik
            </div>
            <div style='font-size:1.4rem; font-weight:800; color:#1B5E20; margin:8px 0;'>
                {best_sc}
            </div>
            <div style='font-size:0.85rem; color:#2E7D32; font-weight:600;'>
                Accuracy: {accs[best_idx]:.2f}% | F1: {f1s[best_idx]:.4f}
            </div>
            <div style='font-size:0.82rem; color:#555; margin-top:10px; line-height:1.6;'>
                Unggul karena representasi sederhana namun efektif untuk Decision Tree.
                Frekuensi integer bekerja optimal dengan threshold-based splitting DT.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#FFF3E0,#FFE0B2);
                    border:2px solid #E65100; border-radius:14px; padding:20px;'>
            <div style='font-size:0.75rem; font-weight:700; color:#E65100;
                        text-transform:uppercase; letter-spacing:0.08em;'>
                🥉 Perlu Peningkatan
            </div>
            <div style='font-size:1.4rem; font-weight:800; color:#BF360C; margin:8px 0;'>
                {worst_sc}
            </div>
            <div style='font-size:0.85rem; color:#E65100; font-weight:600;'>
                Accuracy: {accs[worst_idx]:.2f}% | F1: {f1s[worst_idx]:.4f}
            </div>
            <div style='font-size:0.82rem; color:#555; margin-top:10px; line-height:1.6;'>
                Sparse matrix bigram/trigram menyulitkan DT menemukan threshold optimal.
                Data 10k baris tidak cukup untuk bigram bekerja optimal.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#E3F2FD,#BBDEFB);
                    border:2px solid #1565C0; border-radius:14px; padding:20px;'>
            <div style='font-size:0.75rem; font-weight:700; color:#1565C0;
                        text-transform:uppercase; letter-spacing:0.08em;'>
                💡 Insight Utama
            </div>
            <div style='font-size:0.85rem; color:#0D47A1; font-weight:600; margin:8px 0;'>
                Selisih BoW vs TF-IDF hanya {abs(accs[0]-accs[2]):.2f}%
            </div>
            <div style='font-size:0.82rem; color:#555; margin-top:4px; line-height:1.6;'>
                DT tidak sensitif terhadap skala nilai — pembobotan TF-IDF tidak memberikan
                keunggulan signifikan vs frekuensi integer BoW untuk algoritma ini.
                TF-IDF lebih unggul pada model linier (LR, SVM).
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HALAMAN 7: PREDIKSI INTERAKTIF
# ════════════════════════════════════════════════════════════════
def page_prediction(models, vectorizers, results, le, best_name):
    st.markdown("## 🎯 Prediksi Interaktif")
    st.markdown("Masukkan pertanyaan petani untuk diklasifikasikan secara real-time.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Info model terbaik
    best_r = results[best_name]
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#E8F5E9,#C8E6C9);
                border:2px solid #2E7D32; border-radius:14px; padding:18px 24px;
                margin-bottom:24px; display:flex; align-items:center; gap:20px;'>
        <div style='font-size:2.5rem;'>🤖</div>
        <div>
            <div style='font-size:0.75rem; font-weight:700; color:#2E7D32;
                        text-transform:uppercase; letter-spacing:0.08em;'>
                Model Aktif (Terbaik)
            </div>
            <div style='font-size:1.2rem; font-weight:800; color:#1B5E20;'>
                {best_name}
            </div>
            <div style='font-size:0.85rem; color:#2E7D32; margin-top:2px;'>
                Accuracy: {best_r["accuracy"]*100:.2f}% &nbsp;|&nbsp;
                F1-Score: {best_r["f1_score"]:.4f} &nbsp;|&nbsp;
                Decision Tree (gini, depth=20)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input
    col_inp, col_ex = st.columns([2, 1])

    with col_inp:
        st.markdown("**✏️ Masukkan pertanyaan petani:**")
        user_query = st.text_area("", height=130,
            placeholder="Contoh: My paddy leaves are turning yellow and I see small insects...",
            key="pred_input", label_visibility="collapsed")

        col_sc, col_btn = st.columns([1.5, 1])
        with col_sc:
            sel_sc = st.selectbox("🔧 Pilih Skenario:", SCENARIO_NAMES,
                                   index=SCENARIO_NAMES.index(best_name))
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True)

    with col_ex:
        st.markdown("**📌 Contoh Pertanyaan:**")
        examples_agri  = [
            "My paddy leaves are turning yellow, what should I do?",
            "What is the recommended dose of urea for wheat per acre?",
            "How to control stem borer in rice during kharif season?",
        ]
        examples_horti = [
            "How to improve fruit size and quality in mango orchard?",
            "What pesticide for powdery mildew in chilli plants?",
            "Best grafting technique for guava seedlings?",
        ]
        with st.expander("🌾 Agriculture"):
            for ex in examples_agri:
                st.caption(f"• {ex}")
        with st.expander("🌿 Horticulture"):
            for ex in examples_horti:
                st.caption(f"• {ex}")

    # Prediksi
    if predict_btn and user_query.strip():
        model = models[sel_sc]
        vec   = vectorizers[sel_sc]

        with st.spinner("🔄 Memproses prediksi..."):
            time.sleep(0.4)

            # Pipeline
            clean = clean_text(user_query)
            vector = vec.transform([clean])
            pred_label = model.predict(vector)[0]
            pred_sector = le.inverse_transform([pred_label])[0]

            try:
                proba = model.predict_proba(vector)[0]
                conf  = proba[pred_label] * 100
                conf_str = f"{conf:.1f}%"
            except Exception:
                conf_str = "N/A"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏆 Hasil Prediksi")
        st.markdown("<br>", unsafe_allow_html=True)

        col_r, col_proc = st.columns([1, 1.4])

        with col_r:
            emoji   = "🌾" if pred_sector == "AGRICULTURE" else "🌿"
            css_cls = "pred-result-agri" if pred_sector == "AGRICULTURE" else "pred-result-horti"
            bg_col  = "#2E7D32" if pred_sector == "AGRICULTURE" else "#558B2F"
            st.markdown(f"""
            <div class='{css_cls}'>
                <div style='font-size:3.5rem;'>{emoji}</div>
                <div style='font-size:0.78rem; font-weight:700; color:#2E7D32;
                            text-transform:uppercase; letter-spacing:0.1em; margin:8px 0 4px;'>
                    Sektor Terdeteksi
                </div>
                <div class='pred-label' style='color:{bg_col};'>{pred_sector}</div>
                <div class='pred-conf'>Confidence: <strong>{conf_str}</strong></div>
                <div style='margin-top:12px; font-size:0.82rem; color:#555;'>
                    Model: {sel_sc} | DT (gini, depth=20)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_proc:
            st.markdown("""
            <div class='nlp-card'>
                <h4>🔄 Pipeline Prediksi</h4>
            </div>
            """, unsafe_allow_html=True)

            steps_pred = preprocessing_steps(user_query)
            pipeline_steps = [
                ("1. Input Asli",         user_query[:100] + "..." if len(user_query) > 100 else user_query),
                ("2. Lowercase",          steps_pred["lowercase"][:100]),
                ("3. Hapus Tanda Baca",   steps_pred["no_punct"][:100]),
                ("4. Stopword Removal",   steps_pred["clean"][:100]),
                ("5. Vectorize",          f"CountVectorizer/TfidfVectorizer → vektor sparse"),
                ("6. model.predict()",    f"DecisionTreeClassifier → label {pred_label}"),
                ("7. Hasil Akhir",        f"{emoji} {pred_sector} ({conf_str})"),
            ]
            for label, val in pipeline_steps:
                color = "#E8F5E9" if "Hasil" in label else "#FAFAFA"
                bord  = "#2E7D32" if "Hasil" in label else "#E0E0E0"
                st.markdown(f"""
                <div style='background:{color}; border:1.5px solid {bord};
                            border-radius:8px; padding:8px 12px; margin:4px 0;'>
                    <span style='font-size:0.72rem; font-weight:700; color:#2E7D32;'>{label}:</span>
                    <span style='font-size:0.82rem; color:#333; font-family:"JetBrains Mono",monospace;'>
                        {val}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # Prediksi 3 skenario sekaligus
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔄 Perbandingan Prediksi — 3 Skenario")
        cols_sc = st.columns(3)
        for ci, sname in enumerate(SCENARIO_NAMES):
            m = models[sname]
            v = vectorizers[sname]
            cl = clean_text(user_query)
            vv = v.transform([cl])
            pl = m.predict(vv)[0]
            ps = le.inverse_transform([pl])[0]
            try:
                pb = m.predict_proba(vv)[0]
                cs = f"{pb[pl]*100:.1f}%"
            except Exception:
                cs = "N/A"
            em = "🌾" if ps == "AGRICULTURE" else "🌿"
            bg = "#E8F5E9" if ps == "AGRICULTURE" else "#F1F8E9"
            bd = "#2E7D32" if ps == "AGRICULTURE" else "#558B2F"
            best_tag = " 🏅" if sname == best_name else ""
            cols_sc[ci].markdown(f"""
            <div style='background:{bg}; border:2px solid {bd}; border-radius:12px;
                        padding:18px; text-align:center;'>
                <div style='font-size:0.78rem; font-weight:700; color:{bd};
                            text-transform:uppercase; letter-spacing:0.06em;'>
                    {sname}{best_tag}
                </div>
                <div style='font-size:2rem; margin:8px 0;'>{em}</div>
                <div style='font-size:1rem; font-weight:800; color:{bd};'>{ps}</div>
                <div style='font-size:0.82rem; color:#555; margin-top:4px;'>
                    Conf: {cs}
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif predict_btn and not user_query.strip():
        st.warning("⚠️ Masukkan teks pertanyaan petani terlebih dahulu!")


def render_footer():
    """Render footer di bagian bawah halaman."""
    st.markdown("""
    <div class='dashboard-footer'>
        <div style='margin-bottom:6px;'>
            🌾 <strong>NLP Dashboard</strong> — Klasifikasi Pertanyaan Petani
        </div>
        <div>
            Mini-Project Natural Language Processing · Decision Tree + BoW / N-Gram / TF-IDF
        </div>
        <div style='margin-top:4px; font-size:0.75rem; opacity:0.7;'>
            Dataset: Kisan Call Centre (KCC) India 2013–2021 · Powered by Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MAIN CONTROLLER
# ════════════════════════════════════════════════════════════════
def main():
    menu, uploaded, has_local, local_csv = render_sidebar()

    # ── Load data ─────────────────────────────────────────────────
    df_raw = None

    # Prioritas: uploaded file > local file
    if uploaded is not None:
        try:
            df_raw = load_data_from_upload(uploaded)
            st.session_state["data_source"] = "upload"
        except Exception as e:
            st.error(f"❌ Gagal membaca file upload: {e}")
    elif has_local:
        try:
            df_raw = load_data(local_csv)
            st.session_state["data_source"] = "local"
        except Exception as e:
            st.error(f"❌ Gagal membaca file lokal: {e}")

    # ── Halaman tanpa data ────────────────────────────────────────
    if "🏠" in menu:
        page_home()
        if df_raw is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            src = st.session_state.get("data_source", "unknown")
            src_text = "📂 file lokal" if src == "local" else "📤 file upload"
            st.success(f"✅ Dataset berhasil dimuat dari {src_text} — **{len(df_raw):,} baris** siap diproses!")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 **Upload file `query_agg.csv`** di sidebar atau letakkan di folder yang sama dengan dashboard!")
        render_footer()
        return

    # ── Perlu data untuk halaman lainnya ─────────────────────────
    if df_raw is None:
        st.markdown("""
        <div style='text-align:center; padding:100px 40px;'>
            <div style='font-size:5rem; margin-bottom:20px; filter:drop-shadow(0 4px 12px rgba(0,0,0,0.1));'>📁</div>
            <div style='font-size:1.6rem; font-weight:800; color:#1B5E20; margin-bottom:14px;'>
                Dataset Belum Tersedia
            </div>
            <div style='color:#555; font-size:1rem; max-width:520px; margin:0 auto; line-height:1.7;'>
                Letakkan file <code>query_agg.csv</code> di folder yang sama dengan dashboard,
                atau upload melalui panel sidebar untuk mengaktifkan semua fitur.
            </div>
            <div style='margin-top:28px; color:#888; font-size:0.88rem;'>
                🔗 <a href='https://www.kaggle.com/datasets/anirudhvadakedath/kisan-query-analysis-dataset'
                      target='_blank' style='color:#2E7D32; font-weight:600;'>Download dari Kaggle</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_footer()
        return

    # ── Prepare data & train ──────────────────────────────────────
    try:
        with st.spinner("⚙️ Mempersiapkan data dan melatih model..."):
            df_clean, le = prepare_data(df_raw)
            (models, vectorizers, results, predictions, train_times,
             y_test, le, best_name, best_model, best_vec,
             X_train, X_test, y_train) = train_all_models(df_clean, le)
    except Exception as e:
        st.error(f"❌ Error saat memproses data: {e}")
        st.info("💡 Pastikan file CSV memiliki kolom `QueryText` dan `Sector`.")
        render_footer()
        return

    # ── Routing halaman ───────────────────────────────────────────
    try:
        if   "📂" in menu: page_dataset(df_raw, df_clean)
        elif "🧹" in menu: page_preprocessing(df_clean)
        elif "⚙️" in menu: page_feature_extraction(vectorizers, df_clean)
        elif "🤖" in menu: page_training(results, train_times, predictions, y_test, le)
        elif "📊" in menu: page_comparison(results, train_times)
        elif "🎯" in menu: page_prediction(models, vectorizers, results, le, best_name)
    except Exception as e:
        st.error(f"❌ Terjadi error pada halaman ini: {e}")
        st.info("💡 Silakan coba halaman lain atau reload dashboard.")

    render_footer()


if __name__ == "__main__":
    main()


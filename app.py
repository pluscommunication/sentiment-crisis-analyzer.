import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

# ---- MODULE INTERNE ----
from utils import (
    load_models,
    smart_read_excel,
    run_pipeline_with_progress,
    compute_daily_features,
    detect_crisis_regimes,
    to_excel_bytes,
)

# -------------------------------
#   CONFIGURARE STRATEGICĂ UI
# -------------------------------

st.set_page_config(
    page_title="Sentiment & Crisis Analyzer",
    page_icon="📊",
    layout="wide"
)

# Importăm stilul personalizat
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------
#      HERO SECTION (Header)
# -------------------------------

col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown(
        """
        <div class="hero-title">Sentiment & Crisis Analyzer</div>
        <p class="hero-subtitle">
            Dashboard avansat pentru analiza tonului, emoțiilor și identificarea automată
            a momentelor de criză în comunicarea instituțională.
        </p>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        """
        <div class="kpi-card" style="margin-top:8px;">
            <div class="kpi-label">Status</div>
            <div class="kpi-value" style="font-size:1.1rem;">Transformers active</div>
            <div style="color:#6b7280;font-size:0.8rem;margin-top:4px;">
                Modele încărcate automat pe CPU (compatibil Streamlit Cloud)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="main-block">', unsafe_allow_html=True)

# -------------------------------
#           SIDEBAR
# -------------------------------

with st.sidebar:
    st.header("⚙️ Setări analiză")
    uploaded_file = st.file_uploader(
        "Încarcă fișier Excel",
        type=["xlsx"]
    )
    batch_size = st.slider(
        "Batch size (viteză analiză)", 4, 32, 16, step=4
    )
    crisis_threshold = st.slider(
        "Prag scor criză", 0.5, 2.5, 1.2, step=0.1
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#555;'>Analiza rulează pe CPU. Fișiere mari pot dura câteva minute.</small>",
        unsafe_allow_html=True
    )

# -------------------------------
#     ÎNCĂRCARE FIȘIER
# -------------------------------

if uploaded_file is None:
    st.info("Încarcă un fișier Excel pentru a începe analiza.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

with st.spinner("Se analizează structura fișierului..."):
    df_raw = smart_read_excel(uploaded_file)

st.success("Fișier încărcat cu succes.")

st.markdown("### 📄 Preview date")
st.dataframe(df_raw.head(10), use_container_width=True)

cols = df_raw.columns.tolist()

st.markdown("### 🧩 Selectează coloanele relevante")
col1, col2 = st.columns(2)
with col1:
    text_col = st.selectbox(
        "Coloana text",
        cols,
        index=cols.index("Message") if "Message" in cols else 0
    )
with col2:
    date_col = st.selectbox(
        "Coloana dată",
        cols,
        index=cols.index("Date") if "Date" in cols else 0
    )

st.markdown("---")

# -------------------------------
#    BUTON ANALIZĂ COMPLETĂ
# -------------------------------

run_button = st.button("🚀 Rulează analiza completă")

if run_button or "results" not in st.session_state:

    # 1. Încărcăm modelele
    with st.spinner("Se încarcă modelele NLP..."):
        sent_pipe, emo_pipe = load_models()

    # 2. Pregătim datele
    df = df_raw.copy()
    df["text"] = df[text_col].astype(str).fillna("")
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Filtrare rânduri fără dată sau fără text
    df = df[df["text"].str.strip().ne("")]
    df = df.dropna(subset=["Date"])

    texts = df["text"].tolist()

    # 3. Rulăm analizele NLP cu progres
    st.markdown("### 🔄 Analiză NLP în desfășurare")
    progress_placeholder = st.empty()

    # SENTIMENT
    sent_outputs = run_pipeline_with_progress(
        sent_pipe, texts, batch_size, "Sentiment", progress_placeholder
    )

    # EMOȚII
    emo_outputs = run_pipeline_with_progress(
        emo_pipe, texts, batch_size, "Emoții", progress_placeholder
    )

    st.success("Modelele NLP au rulat cu succes.")

    # 4. Atașăm rezultatele NLP la DataFrame
    df["sentiment_raw"] = [
        o[0]["label"] if isinstance(o, list) and len(o) > 0 else o["label"]
        for o in sent_outputs
    ]
    df["sentiment_score"] = [
        o[0]["score"] if isinstance(o, list) and len(o) > 0 else o["score"]
        for o in sent_outputs
    ]
    df["emotion"] = [
        o[0]["label"] if isinstance(o, list) and len(o) > 0 else o["label"]
        for o in emo_outputs
    ]
    df["emotion_score"] = [
        o[0]["score"] if isinstance(o, list) and len(o) > 0 else o["score"]
        for o in emo_outputs
    ]

    # Mapare label-uri pentru sentiment
    map_sent = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        "NEGATIVE": "negative",
        "NEUTRAL": "neutral",
        "POSITIVE": "positive",
    }
    df["sentiment"] = df["sentiment_raw"].replace(map_sent)

    # 5. DAILY FEATURES (serii zilnice)
    df_daily = compute_daily_features(df)

    # 6. Regimuri și crize (zile de criză + change points)
    df_daily, cluster_profile, crisis_days, change_days = detect_crisis_regimes(
        df_daily, crisis_threshold=crisis_threshold
    )

    # 7. Postări critice (din zilele de criză)
    crisis_dates = crisis_days["Day"].dt.date.tolist()
    df_critical = df[df["Date"].dt.date.isin(crisis_dates)].copy()

    # 8. Salvăm în sesiune
    st.session_state["results"] = {
        "df_posts": df,
        "df_daily": df_daily,
        "df_critical": df_critical,
        "change_days": change_days,
        "crisis_days": crisis_days,
    }

# -------------------------------
#        REZULTATE
# -------------------------------

results = st.session_state["results"]
df_posts = results["df_posts"]
df_daily = results["df_daily"]
df_critical = results["df_critical"]
change_days = results["change_days"]
crisis_days = results["crisis_days"]

# -------------------------------
#        KPI SECTION
# -------------------------------

st.markdown("## 🔍 Rezumat general")

total_posts = len(df_posts)
pos_pct = (df_posts["sentiment"] == "positive").mean() * 100
neg_pct = (df_posts["sentiment"] == "negative").mean() * 100
dom_emotion = df_posts["emotion"].value_counts().idxmax() if not df_posts["emotion"].isna().all() else "N/A"
n_crisis_days = df_daily["is_crisis"].sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Postări analizate</div>
            <div class="kpi-value">{total_posts:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">% pozitive</div>
            <div class="kpi-value">{pos_pct:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">% negative</div>
            <div class="kpi-value">{neg_pct:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Zile de criză</div>
            <div class="kpi-value">{n_crisis_days}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# -------------------------------
#            TABS
# -------------------------------

tab_overview, tab_trends, tab_crisis, tab_emotions, tab_data = st.tabs(
    ["🌐 Overview", "⏱️ Time series", "🚨 Crize", "🎭 Emoții", "📁 Export"]
)

# -------------------------------------------
#  TAB OVERVIEW – Distribuții globale
# -------------------------------------------

with tab_overview:
    st.markdown("### 📊 Distribuții globale")

    colA, colB = st.columns(2)

    with colA:
        sent = df_posts["sentiment"].value_counts().reset_index()
        sent.columns = ["sentiment", "count"]

        fig = px.bar(
            sent,
            x="sentiment",
            y="count",
            color="sentiment",
            title="Distribuția sentimentului",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        emo = df_posts["emotion"].value_counts().reset_index()
        emo.columns = ["emotion", "count"]

        fig2 = px.bar(
            emo,
            x="emotion",
            y="count",
            color="emotion",
            title="Distribuția emoțiilor",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------
#  TAB TRENDS – Time series
# -------------------------------------------

with tab_trends:
    st.markdown("### ⏱️ Evoluția tonului în timp")

    fig_ts = go.Figure()

    fig_ts.add_trace(
        go.Scatter(
            x=df_daily["Day"],
            y=df_daily["sent_mean"],
            mode="lines",
            name="Sentiment mediu",
            line=dict(color="#2563eb", width=2.2),
        )
    )

    # Marcare crize
    crisis_points = df_daily[df_daily["is_crisis"]]
    fig_ts.add_trace(
        go.Scatter(
            x=crisis_points["Day"],
            y=crisis_points["sent_mean"],
            mode="markers",
            marker=dict(size=10, color="#ef4444"),
            name="Zile de criză",
        )
    )

    fig_ts.update_layout(
        title="Sentiment mediu pe zi",
        yaxis_title="(-1) negativ / (+1) pozitiv",
    )

    st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------------------------
#  TAB CRISIS – analiza crizelor
# -------------------------------------------

with tab_crisis:
    st.markdown("### 🚨 Zile de criză detectate")
    st.dataframe(
        df_daily[df_daily["is_crisis"]],
        use_container_width=True,
    )

    st.markdown("### 📌 Postări din zile critice")
    st.dataframe(
        df_critical[
            ["Date", text_col, "sentiment", "sentiment_score", "emotion", "emotion_score"]
        ],
        use_container_width=True,
    )

# -------------------------------------------
#  TAB EMOTIONS – heatmap & metrics
# -------------------------------------------

with tab_emotions:
    st.markdown("### 🎭 Mapa termică a emoțiilor")

    emo_map = df_daily[
        ["anger_ratio", "joy_ratio", "sadness_ratio", "fear_ratio"]
    ].T

    fig_heat = px.imshow(
        emo_map,
        labels=dict(x="Data", y="Emoție", color="Proporție"),
        x=df_daily["Day"],
        y=["Furie", "Bucurie", "Tristețe", "Teamă"],
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# -------------------------------------------
#  TAB EXPORT – Excel downloads
# -------------------------------------------

with tab_data:
    st.markdown("### 📁 Descarcă rezultatele")

    colE1, colE2, colE3 = st.columns(3)

    with colE1:
        st.download_button(
            "📄 Postări (+sentiment + emoții)",
            to_excel_bytes(df_posts),
            file_name="postari_sentiment_emo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="d1",
            type="primary",
        )

    with colE2:
        st.download_button(
            "📈 Serii zilnice + crize",
            to_excel_bytes(df_daily),
            file_name="serii_zilnice_criza.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="d2",
            type="primary",
        )

    with colE3:
        st.download_button(
            "🚨 Postări din crize",
            to_excel_bytes(df_critical),
            file_name="postari_crize.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="d3",
            type="primary",
        )

# Închidem blocul vizual principal
st.markdown("</div>", unsafe_allow_html=True)

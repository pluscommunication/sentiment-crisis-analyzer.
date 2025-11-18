import warnings
warnings.filterwarnings("ignore")

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd

import streamlit as st
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import ruptures as rpt
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# CONFIGURARE PAGINĂ & STIL
# ============================================

st.set_page_config(
    page_title="Sentiment & Crisis Analyzer",
    page_icon="📊",
    layout="wide"
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #020617 0, #020617 30%, #000000 100%);
        color: #e5e7eb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: 0.04em;
        color: #e5e7eb !important;
    }
    .kpi-card {
        padding: 1rem 1.5rem;
        border-radius: 1.2rem;
        background: radial-gradient(circle at top left, #0f172a, #020617);
        border: 1px solid rgba(148, 163, 184, 0.45);
        box-shadow: 0 22px 60px rgba(15, 23, 42, 0.95);
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9ca3af;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #020617);
        border-right: 1px solid rgba(148, 163, 184, 0.4);
    }
    .stDataFrame {
        border-radius: 1rem;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.6rem 1.5rem;
        border: none;
        background: linear-gradient(135deg, #4f46e5, #22c55e);
        color: white;
        font-weight: 600;
        letter-spacing: 0.06em;
        box-shadow: 0 14px 35px rgba(34, 197, 94, 0.55);
    }
    .stButton>button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================
# FUNCȚII UTILITARE
# ============================================

@st.cache_resource
def load_models():
    """Încarcă și cache-uiește modelele Transformer."""
    sentiment_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    emotion_model = "j-hartmann/emotion-english-distilroberta-base"

    sent_pipe = pipeline(
        "sentiment-analysis",
        model=sentiment_model,
        tokenizer=sentiment_model,
        truncation=True,
        max_length=512,
    )

    emo_pipe = pipeline(
        "text-classification",
        model=emotion_model,
        tokenizer=emotion_model,
        top_k=1,
        truncation=True,
        max_length=512,
    )

    return sent_pipe, emo_pipe


def smart_read_excel(uploaded_file):
    """
    Încarcă un Excel Crowdtangle-like și detectează automat header-ul
    cu 'Date' și 'Message'. Dacă nu găsește, folosește primul rând ca header.
    """
    raw = pd.read_excel(uploaded_file, header=None, engine="openpyxl")

    header_row_idx = None
    for i, row in raw.iterrows():
        vals = row.astype(str).tolist()
        if "Date" in vals and "Message" in vals:
            header_row_idx = i
            break

    if header_row_idx is None:
        df = pd.read_excel(uploaded_file)
    else:
        header = raw.iloc[header_row_idx].tolist()
        df = raw.iloc[header_row_idx + 1 :].copy()
        df.columns = pd.Index(header).astype(str).str.strip()
        df = df.dropna(axis=1, how="all")

    return df


def run_pipeline_with_progress(pipe, texts, batch_size, label, progress_container):
    """Rulează un pipeline Transformers pe loturi, cu bară de progres în UI."""
    total = len(texts)
    if total == 0:
        return []
    num_batches = math.ceil(total / batch_size)
    results = []
    progress_bar = progress_container.progress(0.0, text=f"{label}: 0%")

    for i in range(num_batches):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        outputs = pipe(batch)
        results.extend(outputs)
        progress = (i + 1) / num_batches
        progress_bar.progress(progress, text=f"{label}: {int(progress*100)}%")

    return results


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Transformă un DataFrame în bytes pentru download Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


# ============================================
# ANALIZĂ COMPLETĂ: SENTIMENT + EMOȚII + CRIZE
# ============================================

def full_advanced_analysis(df, text_col, date_col, batch_size=16, crisis_threshold=1.2):
    """Rulează tot pipeline-ul avansat și returnează structuri gata de utilizare."""

    df = df.copy()
    df["text"] = df[text_col].astype(str).fillna("")
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df["Date"].dt.year

    # Filtrăm rândurile fără text sau fără dată
    df = df[df["text"].str.strip().ne("")]
    df = df.dropna(subset=["Date"])

    # === încărcăm modelele ===
    sent_pipe, emo_pipe = load_models()

    # === rulăm modele ===
    texts = df["text"].tolist()

    progress_holder = st.empty()
    with st.spinner("Se rulează analiza de sentiment și emoții..."):
        sent_outputs = run_pipeline_with_progress(
            sent_pipe, texts, batch_size, "Analiză sentiment", progress_holder
        )
        emo_outputs = run_pipeline_with_progress(
            emo_pipe, texts, batch_size, "Analiză emoții", progress_holder
        )

    # === procesăm output sentiment ===
    sent_labels = []
    sent_scores = []
    for o in sent_outputs:
        if isinstance(o, list) and len(o) > 0:
            o = o[0]
        label = o.get("label")
        score = o.get("score")
        sent_labels.append(label)
        sent_scores.append(score)

    df["sentiment_raw"] = sent_labels
    df["sentiment_score"] = sent_scores

    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        "NEGATIVE": "negative",
        "NEUTRAL": "neutral",
        "POSITIVE": "positive",
    }
    df["sentiment"] = df["sentiment_raw"].replace(label_map)

    # === procesăm output emoții ===
    emo_labels = []
    emo_scores = []
    for r in emo_outputs:
        if isinstance(r, list) and len(r) > 0:
            r = r[0]
        label = r.get("label") if isinstance(r, dict) else None
        score = r.get("score") if isinstance(r, dict) else None
        emo_labels.append(label)
        emo_scores.append(score)

    df["emotion"] = emo_labels
    df["emotion_score"] = emo_scores

    # === FEATURES ZILNICE ===
    df_ts = df.copy()

    sent_map = {"negative": -1, "neutral": 0, "positive": 1}
    df_ts["sentiment_num"] = df_ts["sentiment"].map(sent_map)

    def emotion_ratio(group, emo):
        return (group["emotion"] == emo).sum() / len(group) if len(group) > 0 else 0

    daily = (
        df_ts
        .groupby(df_ts["Date"].dt.date)
        .apply(lambda g: pd.Series({
            "n_posts": len(g),
            "sent_mean": g["sentiment_num"].mean(),
            "sent_std": g["sentiment_num"].std(ddof=0) if len(g) > 1 else 0,
            "neg_ratio": (g["sentiment"] == "negative").mean(),
            "pos_ratio": (g["sentiment"] == "positive").mean(),
            "anger_ratio": emotion_ratio(g, "anger"),
            "joy_ratio": emotion_ratio(g, "joy"),
            "sadness_ratio": emotion_ratio(g, "sadness"),
            "fear_ratio": emotion_ratio(g, "fear"),
        }))
        .reset_index()
        .rename(columns={"Date": "Day"})
    )

    daily["Day"] = pd.to_datetime(daily["Day"], errors="coerce")
    daily = daily.sort_values("Day").reset_index(drop=True)

    # === SCOR DE CRIZĂ – neg_ratio + sent_std + n_posts (z-score pozitivi) ===
    for col in ["neg_ratio", "sent_std", "n_posts"]:
        m = daily[col].mean()
        s = daily[col].std(ddof=0) if daily[col].std(ddof=0) > 0 else 1.0
        daily[f"{col}_z"] = (daily[col] - m) / s

    daily["neg_z_pos"] = daily["neg_ratio_z"].clip(lower=0)
    daily["std_z_pos"] = daily["sent_std_z"].clip(lower=0)
    daily["vol_z_pos"] = daily["n_posts_z"].clip(lower=0)

    daily["crisis_score"] = (
        daily["neg_z_pos"] +
        daily["std_z_pos"] +
        daily["vol_z_pos"]
    ) / 3.0

    daily["is_crisis"] = (daily["crisis_score"] > crisis_threshold) & (daily["n_posts"] >= 3)

    # === CLUSTERING (opțional – pentru analiză) ===
    feature_cols = [
        "n_posts",
        "sent_mean",
        "sent_std",
        "neg_ratio",
        "pos_ratio",
        "anger_ratio",
        "joy_ratio",
        "sadness_ratio",
        "fear_ratio",
    ]
    X = daily[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    daily["regime_cluster"] = kmeans.fit_predict(X_scaled)
    cluster_profile = daily.groupby("regime_cluster")[feature_cols].mean()

    regime_names = {0: "Regim 0", 1: "Regim 1", 2: "Regim 2"}
    daily["regime_name"] = daily["regime_cluster"].map(regime_names)

    # === CHANGE POINTS PE SENT_MEAN (rupturi de ton) ===
    signal = daily["sent_mean"].fillna(0).values
    algo = rpt.Pelt(model="rbf").fit(signal)
    change_indices = algo.predict(pen=5)

    change_days = daily.loc[
        [i - 1 for i in change_indices if 0 <= i - 1 < len(daily)],
        "Day"
    ]

    crisis_days = daily[daily["is_crisis"]].copy()

    # === POSTĂRI DIN ZILELE DE CRIZĂ ===
    crisis_dates = crisis_days["Day"].dt.date.tolist()
    df_critical = df_ts[df_ts["Date"].dt.date.isin(crisis_dates)].copy()
    df_critical["sentiment_num"] = df_critical["sentiment"].map(sent_map)
    df_critical = df_critical.sort_values(
        by=["sentiment_num", "sentiment_score"],
        ascending=[True, True]
    )

    return {
        "df_posts": df,
        "df_daily": daily,
        "df_critical": df_critical,
        "cluster_profile": cluster_profile,
        "change_days": change_days,
        "crisis_days": crisis_days,
    }


# ============================================
# UI – APLICAȚIA STREAMLIT
# ============================================

st.title("📊 Sentiment & Crisis Analyzer")
st.subheader("Analiză avansată de ton, emoții și crize în comunicarea instituțională")

st.markdown(
    """
    <p style="color:#9ca3af; max-width:900px;">
    Încarcă un fișier Excel cu postările unei instituții (de exemplu Ministerul Economiei).
    Aplicația va rula modele Transformers pentru a estima sentimentul și emoțiile,
    va construi serii de timp, va calcula un scor compus de criză și va identifica
    zilele critice, evidențiate vizual în grafice.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Setări analiză")
    uploaded_file = st.file_uploader(
        "Trage și plasează fișierul Excel aici",
        type=["xlsx"],
        accept_multiple_files=False
    )
    batch_size = st.slider(
        "Batch size pentru modele (mai mare = mai rapid, dar mai greu pentru CPU)",
        min_value=4,
        max_value=32,
        value=16,
        step=4
    )
    crisis_threshold = st.slider(
        "Prag scor de criză (mai mic = detectează mai multe crize)",
        min_value=0.5,
        max_value=2.5,
        value=1.2,
        step=0.1
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#9ca3af;'>Analiza se face local, pe CPU. Pentru fișiere mari, poate dura câteva minute.</small>",
        unsafe_allow_html=True
    )

if uploaded_file is None:
    st.info("Încarcă un fișier Excel pentru a începe analiza.")
    st.stop()

# 1. ÎNCĂRCARE & PREVIEW
with st.spinner("Se încarcă fișierul și se detectează structura..."):
    df_raw = smart_read_excel(uploaded_file)

st.success("Fișier încărcat cu succes.")
st.markdown("### 📄 Preview date încărcate")
st.dataframe(df_raw.head(10), use_container_width=True)

# 2. ALEGERE COLOANE
st.markdown("### 🧩 Selectează coloanele relevante")

cols = df_raw.columns.tolist()
col1, col2 = st.columns(2)

with col1:
    text_col = st.selectbox(
        "Coloana cu textul postărilor",
        options=cols,
        index=cols.index("Message") if "Message" in cols else 0
    )

with col2:
    date_col = st.selectbox(
        "Coloana cu data postărilor",
        options=cols,
        index=cols.index("Date") if "Date" in cols else 0
    )

st.markdown("---")

run_button = st.button("🚀 Rulează analiza completă")

if not run_button:
    st.warning("Apasă pe „🚀 Rulează analiza completă” pentru a porni modelele și graficele.")
    st.stop()

# 3. RULĂM ANALIZA COMPLETĂ
results = full_advanced_analysis(
    df_raw,
    text_col,
    date_col,
    batch_size=batch_size,
    crisis_threshold=crisis_threshold
)
df_posts = results["df_posts"]
df_daily = results["df_daily"]
df_critical = results["df_critical"]
cluster_profile = results["cluster_profile"]
change_days = results["change_days"]
crisis_days = results["crisis_days"]

# ==========================
# KPI – METRICE
# ==========================
st.markdown("## 🔍 Rezumat general")

total_posts = len(df_posts)
pos_pct = (df_posts["sentiment"] == "positive").mean() * 100
neg_pct = (df_posts["sentiment"] == "negative").mean() * 100
dom_emotion = df_posts["emotion"].value_counts().idxmax() if not df_posts["emotion"].isna().all() else "N/A"
n_crisis_days = df_daily["is_crisis"].sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">Postări analizate</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{total_posts:,}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">% pozitive</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{pos_pct:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">% negative</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{neg_pct:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">Zile de criză detectate</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{n_crisis_days}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================
# DISTRIBUȚII GLOBALE
# ==========================
st.markdown("## 📊 Distribuții globale")

colg1, colg2 = st.columns(2)
with colg1:
    sent_vc = df_posts["sentiment"].value_counts().reset_index(name="count")
    sent_vc.columns = ["sentiment", "count"]
    fig_sent = px.bar(
        sent_vc,
        x="sentiment",
        y="count",
        labels={"sentiment": "Sentiment", "count": "Număr postări"},
        title="Distribuția sentimentului",
        color="sentiment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_white"
    )
    fig_sent.update_layout(
        plot_bgcolor="rgba(15,23,42,0.9)",
        paper_bgcolor="rgba(15,23,42,0)",
        font_color="#e5e7eb"
    )
    st.plotly_chart(fig_sent, use_container_width=True)

with colg2:
    emo_vc = df_posts["emotion"].value_counts().reset_index(name="count")
    emo_vc.columns = ["emotion", "count"]
    fig_emo = px.bar(
        emo_vc,
        x="emotion",
        y="count",
        labels={"emotion": "Emoție", "count": "Număr postări"},
        title="Distribuția emoțiilor",
        color="emotion",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white"
    )
    fig_emo.update_layout(
        plot_bgcolor="rgba(15,23,42,0.9)",
        paper_bgcolor="rgba(15,23,42,0)",
        font_color="#e5e7eb"
    )
    st.plotly_chart(fig_emo, use_container_width=True)

# ==========================
# TIME SERIES – TON MEDIU + ZILE DE CRIZĂ ÎNCERCUITE
# ==========================
st.markdown("## ⏱️ Dinamica tonului și zilele de criză")

fig_ts = go.Figure()

fig_ts.add_trace(
    go.Scatter(
        x=df_daily["Day"],
        y=df_daily["sent_mean"],
        mode="lines",
        name="Sentiment mediu",
        line=dict(color="#38bdf8", width=2.5)
    )
)

# marcăm zilele de criză
fig_ts.add_trace(
    go.Scatter(
        x=crisis_days["Day"],
        y=crisis_days["sent_mean"],
        mode="markers",
        name="Zile de criză",
        marker=dict(color="#f97316", size=10, symbol="star"),
    )
)

fig_ts.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)

# cercuri „de mână” în jurul zilelor de criză
shapes = []
for _, row in crisis_days.iterrows():
    x = row["Day"]
    y = row["sent_mean"]
    shapes.append(dict(
        type="circle",
        xref="x",
        yref="y",
        x0=x - pd.Timedelta(days=0.6),
        x1=x + pd.Timedelta(days=0.6),
        y0=y - 0.25,
        y1=y + 0.25,
        line=dict(color="rgba(248,113,113,0.95)", width=2),
    ))
    shapes.append(dict(
        type="circle",
        xref="x",
        yref="y",
        x0=x - pd.Timedelta(days=0.5),
        x1=x + pd.Timedelta(days=0.7),
        y0=y - 0.22,
        y1=y + 0.28,
        line=dict(color="rgba(248,113,113,0.55)", width=1.5),
    ))

fig_ts.update_layout(
    title="Evoluția tonului mediu – zilele de criză evidențiate „de mână”",
    xaxis_title="Data",
    yaxis_title="Sentiment mediu (-1 / 0 / +1)",
    template="plotly_white",
    shapes=shapes,
    plot_bgcolor="rgba(15,23,42,0.95)",
    paper_bgcolor="rgba(15,23,42,0)",
    font_color="#e5e7eb",
    legend=dict(bgcolor="rgba(15,23,42,0.7)")
)
st.plotly_chart(fig_ts, use_container_width=True)

# ==========================
# TIME SERIES – PROCENT NEGATIV + ZONE UMBRITE
# ==========================
st.markdown("## 🌪 Procentul de postări negative – zone de criză")

fig_neg = go.Figure()

fig_neg.add_trace(
    go.Scatter(
        x=df_daily["Day"],
        y=df_daily["neg_ratio"],
        mode="lines+markers",
        name="% postări negative",
        line=dict(color="#fb923c", width=2),
        marker=dict(size=4)
    )
)

for _, row in crisis_days.iterrows():
    d = row["Day"]
    fig_neg.add_vrect(
        x0=d - pd.Timedelta(days=0.5),
        x1=d + pd.Timedelta(days=0.5),
        fillcolor="rgba(248,113,113,0.18)",
        line_width=0,
        layer="below"
    )

mean_neg = df_daily["neg_ratio"].mean()
fig_neg.add_hline(y=mean_neg, line_dash="dot", line_color="gray", annotation_text="Media negativității")

fig_neg.update_layout(
    title="Procentul de postări negative – cu zilele de criză umbrite",
    xaxis_title="Data",
    yaxis_title="Procent postări negative",
    template="plotly_white",
    plot_bgcolor="rgba(15,23,42,0.95)",
    paper_bgcolor="rgba(15,23,42,0)",
    font_color="#e5e7eb"
)
st.plotly_chart(fig_neg, use_container_width=True)

# ==========================
# „HARTA CRIZEI” – VOLUM VS NEGATIVITATE
# ==========================
st.markdown("## 🧭 Harta riscului de criză")

fig_scatter = px.scatter(
    df_daily,
    x="n_posts",
    y="neg_ratio",
    size="crisis_score",
    color="crisis_score",
    color_continuous_scale="Reds",
    hover_data=["Day", "sent_mean", "sent_std"],
    title="Harta riscului de criză: volum de postări vs negativitate",
    labels={
        "n_posts": "Număr postări / zi",
        "neg_ratio": "Procent postări negative",
        "crisis_score": "Scor de criză (compus)"
    },
    template="plotly_white"
)
fig_scatter.update_traces(
    marker=dict(line=dict(width=1, color="rgba(15,23,42,0.9)"))
)
fig_scatter.update_layout(
    plot_bgcolor="rgba(15,23,42,0.95)",
    paper_bgcolor="rgba(15,23,42,0)",
    font_color="#e5e7eb"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================
# HEATMAP EMOȚIONAL
# ==========================
st.markdown("## 🌡️ Mapa termică a emoțiilor în timp")

emo_mat = df_daily[["anger_ratio", "joy_ratio", "sadness_ratio", "fear_ratio"]].T

fig_heat = px.imshow(
    emo_mat,
    labels=dict(x="Data", y="Emoție", color="Proporție"),
    x=df_daily["Day"],
    y=["Furie", "Bucurie", "Tristețe", "Teamă"],
    title="Mapa termică a emoțiilor (pe zile)",
    color_continuous_scale="RdYlBu_r",
    aspect="auto",
    template="plotly_white"
)
for _, row in crisis_days.iterrows():
    d = row["Day"]
    fig_heat.add_vline(x=d, line_dash="dot", line_color="rgba(248,113,113,0.8)", opacity=0.9)

fig_heat.update_layout(
    plot_bgcolor="rgba(15,23,42,0.95)",
    paper_bgcolor="rgba(15,23,42,0)",
    font_color="#e5e7eb"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ==========================
# TIME SERIES – SCOR DE CRIZĂ
# ==========================
st.markdown("## 📈 Scorul compus de criză în timp")

fig_score = px.line(
    df_daily,
    x="Day",
    y="crisis_score",
    title="Scorul compus de criză (negativitate + volatilitate + volum)",
    labels={"Day": "Data", "crisis_score": "Scor de criză"},
    template="plotly_white"
)
fig_score.add_hline(
    y=crisis_threshold,
    line_dash="dash",
    line_color="red",
    annotation_text="Prag criză",
    annotation_position="top right"
)
fig_score.add_trace(
    go.Scatter(
        x=crisis_days["Day"],
        y=crisis_days["crisis_score"],
        mode="markers",
        name="Zile de criză",
        marker=dict(color="#f97316", size=10, symbol="circle-open-dot")
    )
)
fig_score.update_layout(
    plot_bgcolor="rgba(15,23,42,0.95)",
    paper_bgcolor="rgba(15,23,42,0)",
    font_color="#e5e7eb"
)
st.plotly_chart(fig_score, use_container_width=True)

# ==========================
# TABEL ZILE CRITICE & POSTĂRI
# ==========================
st.markdown("## 🚨 Zile critice și postările aferente")

st.markdown(
    "<p style='color:#9ca3af;'>Zilele de criză sunt identificate printr-un scor compus "
    "care combină negativitatea, volatilitatea tonului și volumul de postări.</p>",
    unsafe_allow_html=True
)

st.markdown("### 🗓️ Zile critice (scor ridicat de criză)")
st.dataframe(
    df_daily[df_daily["is_crisis"]][
        ["Day", "n_posts", "sent_mean", "sent_std", "neg_ratio", "crisis_score"]
    ],
    use_container_width=True,
)

st.markdown("### 💬 Exemple de postări din zile critice")
st.dataframe(
    df_critical[[ "Date", text_col, "sentiment", "sentiment_score", "emotion", "emotion_score" ]].head(80),
    use_container_width=True
)

# ==========================
# DOWNLOAD REZULTATE
# ==========================
st.markdown("## 📥 Descarcă rezultatele")

cold1, cold2, cold3 = st.columns(3)

with cold1:
    st.download_button(
        label="📄 Postări + scoruri (Excel)",
        data=to_excel_bytes(df_posts),
        file_name="sentiment_emotii_postari.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with cold2:
    st.download_button(
        label="📈 Serii zilnice + scor criză (Excel)",
        data=to_excel_bytes(df_daily),
        file_name="serii_zilnice_criza.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with cold3:
    st.download_button(
        label="🚨 Postări din zile critice (Excel)",
        data=to_excel_bytes(df_critical),
        file_name="postari_zile_critice.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

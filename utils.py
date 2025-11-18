import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import ruptures as rpt
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================
# 1. MODELE TRANSFORMER
# ============================================

def load_models():
    """
    Încarcă explicit modelele și tokenizer-ele ca să evităm problemele cu 'meta tensors'
    pe Torch + Transformers în mediul Streamlit Cloud.
    """
    sentiment_model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    emotion_model_id = "j-hartmann/emotion-english-distilroberta-base"

    # --- Sentiment ---
    sent_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_id)
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_model_id,
        low_cpu_mem_usage=False,   # forțăm încărcare completă, fără meta
    )

    sent_pipe = pipeline(
        "sentiment-analysis",
        model=sent_model,
        tokenizer=sent_tokenizer,
        device=-1,                # CPU
        truncation=True,
        max_length=512,
    )

    # --- Emoții ---
    emo_tokenizer = AutoTokenizer.from_pretrained(emotion_model_id)
    emo_model = AutoModelForSequenceClassification.from_pretrained(
        emotion_model_id,
        low_cpu_mem_usage=False,
    )

    emo_pipe = pipeline(
        "text-classification",
        model=emo_model,
        tokenizer=emo_tokenizer,
        top_k=1,
        device=-1,
        truncation=True,
        max_length=512,
    )

    return sent_pipe, emo_pipe


# ============================================
# 2. CITIRE INTELIGENTĂ EXCEL
# ============================================

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


# ============================================
# 3. PIPELINE CU PROGRES
# ============================================

def run_pipeline_with_progress(
    pipe,
    texts,
    batch_size,
    label=None,
    progress_container=None,
    callback=None,
):
    """
    Rulează un pipeline Transformers în batch-uri.

    Poate fi folosit în două moduri:
    - din app.py:
        run_pipeline_with_progress(pipe, texts, batch_size, "Sentiment", placeholder)
      -> actualizează o bară de progres Streamlit
    - din cod intern:
        run_pipeline_with_progress(pipe, texts, batch_size, callback=func)
      -> doar apelări de callback(progress)
    """
    total = len(texts)
    if total == 0:
        return []

    num_batches = math.ceil(total / batch_size)
    results = []

    progress_bar = None
    if progress_container is not None:
        progress_bar = progress_container.progress(
            0.0,
            text=f"{label}: 0%" if label else "0%",
        )

    def internal_callback(p):
        if progress_bar is not None:
            percent = int(p * 100)
            progress_bar.progress(
                p,
                text=f"{label}: {percent}%" if label else f"{percent}%",
            )
        if callback is not None:
            callback(p)

    for i in range(num_batches):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        out = pipe(batch)
        results.extend(out)

        progress = (i + 1) / num_batches
        internal_callback(progress)

    return results


# ============================================
# 4. FEATURES ZILNICE (SERII DE TIMP)
# ============================================

def compute_daily_features(df_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Primește df_posts (cu coloane Date, sentiment, emotion) și construiește
    serii de timp la nivel de zi: volum, ton, emoții, scoruri z-normalizate.
    """
    df = df_posts.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    sent_map = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment_num"] = df["sentiment"].map(sent_map)

    def emotion_ratio(group, emo):
        return (group["emotion"] == emo).sum() / len(group) if len(group) > 0 else 0

    daily = (
        df
        .groupby(df["Date"].dt.date)
        .apply(
            lambda g: pd.Series(
                {
                    "n_posts": len(g),
                    "sent_mean": g["sentiment_num"].mean(),
                    "sent_std": g["sentiment_num"].std(ddof=0) if len(g) > 1 else 0,
                    "neg_ratio": (g["sentiment"] == "negative").mean(),
                    "pos_ratio": (g["sentiment"] == "positive").mean(),
                    "anger_ratio": emotion_ratio(g, "anger"),
                    "joy_ratio": emotion_ratio(g, "joy"),
                    "sadness_ratio": emotion_ratio(g, "sadness"),
                    "fear_ratio": emotion_ratio(g, "fear"),
                }
            )
        )
        .reset_index()
        .rename(columns={"Date": "Day"})
    )

    daily["Day"] = pd.to_datetime(daily["Day"], errors="coerce")
    daily = daily.sort_values("Day").reset_index(drop=True)

    # z-score pentru volum, negativitate, volatilitate
    for col in ["neg_ratio", "sent_std", "n_posts"]:
        m = daily[col].mean()
        s = daily[col].std(ddof=0) if daily[col].std(ddof=0) > 0 else 1.0
        daily[f"{col}_z"] = (daily[col] - m) / s

    daily["neg_z_pos"] = daily["neg_ratio_z"].clip(lower=0)
    daily["std_z_pos"] = daily["sent_std_z"].clip(lower=0)
    daily["vol_z_pos"] = daily["n_posts_z"].clip(lower=0)

    daily["crisis_score"] = (
        daily["neg_z_pos"] + daily["std_z_pos"] + daily["vol_z_pos"]
    ) / 3.0

    return daily


# ============================================
# 5. REGIMURI & CRIZE
# ============================================

def detect_crisis_regimes(daily: pd.DataFrame, crisis_threshold: float = 1.2):
    """
    Primește daily features și:
    - marchează zilele de criză (is_crisis)
    - face clustering în 3 regimuri
    - detectează puncte de schimbare de ton
    - extrage zilele de criză și change points
    """
    df_daily = daily.copy()

    df_daily["is_crisis"] = (
        (df_daily["crisis_score"] > crisis_threshold) & (df_daily["n_posts"] >= 3)
    )

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
    X = df_daily[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_daily["regime_cluster"] = kmeans.fit_predict(X_scaled)
    cluster_profile = df_daily.groupby("regime_cluster")[feature_cols].mean()

    regime_names = {0: "Regim 0", 1: "Regim 1", 2: "Regim 2"}
    df_daily["regime_name"] = df_daily["regime_cluster"].map(regime_names)

    signal = df_daily["sent_mean"].fillna(0).values
    algo = rpt.Pelt(model="rbf").fit(signal)
    change_indices = algo.predict(pen=5)

    change_days = df_daily.loc[
        [i - 1 for i in change_indices if 0 <= i - 1 < len(df_daily)],
        "Day",
    ]

    crisis_days = df_daily[df_daily["is_crisis"]].copy()

    return df_daily, cluster_profile, crisis_days, change_days


# ============================================
# 6. EXCEL EXPORT
# ============================================

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Transformă un DataFrame în bytes pentru download Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

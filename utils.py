import io
import math
from datetime import datetime

import pandas as pd
import numpy as np

from transformers import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import ruptures as rpt


# ======================================================
# 1. UTILITAR — ÎNCĂRCARE MODELE (CACHE)
# ======================================================

def load_models():
    """
    Încarcă modelele într-o manieră compatibilă pentru Streamlit Cloud.
    NU folosește modele mari → compatibil cu spațiu limitat.
    """
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


# ======================================================
# 2. CITIRE EXCEL — DETECTEAZĂ HEADER „Date/Message”
# ======================================================

def smart_read_excel(uploaded_file):
    """
    Identifică automat header-ul (potrivit pentru fișiere CrowdTangle).
    Dacă nu detectează „Date” și „Message”, folosește primul rând.
    """
    raw = pd.read_excel(uploaded_file, header=None, engine="openpyxl")

    header_row_idx = None
    for i, row in raw.iterrows():
        values = row.astype(str).tolist()
        if "Date" in values and "Message" in values:
            header_row_idx = i
            break

    if header_row_idx is None:
        df = pd.read_excel(uploaded_file)
    else:
        header = raw.iloc[header_row_idx].tolist()
        df = raw.iloc[header_row_idx + 1:].copy()
        df.columns = pd.Index(header).astype(str).str.strip()
        df = df.dropna(axis=1, how="all")

    return df


# ======================================================
# 3. PIPELINE CU PROGRES (BATCH)
# ======================================================

def run_pipeline_with_progress(pipe, texts, batch_size, callback=None):
    """
    Rulează un pipeline Transformers în batch-uri.
    callback(progress) → procent progres pentru Streamlit.
    """
    total = len(texts)
    if total == 0:
        return []

    num_batches = math.ceil(total / batch_size)
    results = []

    for i in range(num_batches):
        batch = texts[i * batch_size:(i + 1) * batch_size]
        out = pipe(batch)
        results.extend(out)

        if callback:
            callback((i + 1) / num_batches)

    return results


# ======================================================
# 4. EXPORT EXCEL → BYTES
# ======================================================

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


# ======================================================
# 5. FEATURE ENGINEERING COMPLET (DAILY)
# ======================================================

def build_daily_series(df_posts):
    """
    Creează serii zilnice cu:
    - volum
    - sentiment mean/std
    - negative, positive ratio
    - emoții ratio
    """

    df = df_posts.copy()

    sent_map = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment_num"] = df["sentiment"].map(sent_map)

    def emo_ratio(group, emo):
        return (group["emotion"] == emo).mean()

    daily = (
        df.groupby(df["Date"].dt.date)
        .apply(lambda g: pd.Series({
            "n_posts": len(g),
            "sent_mean": g["sentiment_num"].mean(),
            "sent_std": g["sentiment_num"].std(ddof=0) if len(g) > 1 else 0,
            "neg_ratio": (g["sentiment"] == "negative").mean(),
            "pos_ratio": (g["sentiment"] == "positive").mean(),
            "anger_ratio": emo_ratio(g, "anger"),
            "joy_ratio": emo_ratio(g, "joy"),
            "sadness_ratio": emo_ratio(g, "sadness"),
            "fear_ratio": emo_ratio(g, "fear"),
        }))
        .reset_index()
        .rename(columns={"Date": "Day"})
    )

    daily["Day"] = pd.to_datetime(daily["Day"])

    return daily.sort_values("Day").reset_index(drop=True)


# ======================================================
# 6. SCOR COMPLEX DE CRIZĂ (Z-SCORES)
# ======================================================

def compute_crisis_score(daily, threshold=1.2):
    """
    Creează trei indicatori normalizați:
    - negativitate (neg_ratio)
    - volatilitate ton (sent_std)
    - volum (n_posts)

    Scorul de criză = media z-score-urilor pozitive.
    """
    daily = daily.copy()

    for col in ["neg_ratio", "sent_std", "n_posts"]:
        m = daily[col].mean()
        s = daily[col].std(ddof=0)
        if s == 0:
            s = 1.0
        daily[col + "_z"] = (daily[col] - m) / s

    daily["neg_z_pos"] = daily["neg_ratio_z"].clip(lower=0)
    daily["std_z_pos"] = daily["sent_std_z"].clip(lower=0)
    daily["vol_z_pos"] = daily["n_posts_z"].clip(lower=0)

    daily["crisis_score"] = (
        daily["neg_z_pos"] +
        daily["std_z_pos"] +
        daily["vol_z_pos"]
    ) / 3.0

    daily["is_crisis"] = (daily["crisis_score"] > threshold) & (daily["n_posts"] >= 3)

    return daily


# ======================================================
# 7. CLUSTERING (K-MEANS) — 3 REGIMURI DE TONALITATE
# ======================================================

def cluster_daily_regimes(daily):
    daily = daily.copy()

    features = [
        "n_posts", "sent_mean", "sent_std", "neg_ratio",
        "pos_ratio", "anger_ratio", "joy_ratio",
        "sadness_ratio", "fear_ratio"
    ]

    X = daily[features].fillna(0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    daily["regime_cluster"] = km.fit_predict(Xs)

    profile = daily.groupby("regime_cluster")[features].mean()

    name_map = {0: "Regim 0", 1: "Regim 1", 2: "Regim 2"}
    daily["regime_name"] = daily["regime_cluster"].map(name_map)

    return daily, profile


# ======================================================
# 8. CHANGE POINT DETECTION
# ======================================================

def detect_change_points(daily):
    """
    Utilizează PELT + model RBF pentru detectarea
    schimbărilor în tonul mediu (sent_mean).
    """
    signal = daily["sent_mean"].fillna(0).values

    algo = rpt.Pelt(model="rbf").fit(signal)
    change_indices = algo.predict(pen=5)

    return daily.iloc[[i - 1 for i in change_indices if i - 1 < len(daily)]]["Day"]


# ======================================================
# 9. CRITICAL POSTS (ZILE CU CRIZĂ)
# ======================================================

def extract_critical_posts(df_posts, daily):
    crisis_dates = daily[daily["is_crisis"]]["Day"].dt.date.tolist()
    return df_posts[df_posts["Date"].dt.date.isin(crisis_dates)].copy()


# ======================================================
# 10. ROUTĂ PRINCIPALĂ — ANALIZĂ COMPLETĂ
# ======================================================

def full_advanced_analysis(df_raw, text_col, date_col, batch_size, crisis_threshold, progress_callback=None):
    """
    Combină TOATE etapele într-o singură funcție.
    Returnează structurile necesare UI-ului.
    """

    # pregătire
    df = df_raw.copy()
    df["text"] = df[text_col].astype(str).fillna("")
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["text"].str.strip().ne("")]

    # încarcă modele
    sent_pipe, emo_pipe = load_models()

    # rulează
    texts = df["text"].tolist()

    def progress(p):
        if progress_callback:
            progress_callback(p)

    sent_out = run_pipeline_with_progress(sent_pipe, texts, batch_size, callback=progress)
    emo_out = run_pipeline_with_progress(emo_pipe, texts, batch_size, callback=progress)

    # procesare sentiment
    sent_labels = []
    sent_scores = []
    for o in sent_out:
        o = o[0] if isinstance(o, list) else o
        sent_labels.append(o.get("label"))
        sent_scores.append(o.get("score"))

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

    # procesare emoții
    emo_labels = []
    emo_scores = []
    for r in emo_out:
        r = r[0] if isinstance(r, list) else r
        emo_labels.append(r.get("label"))
        emo_scores.append(r.get("score"))
    df["emotion"] = emo_labels
    df["emotion_score"] = emo_scores

    # FEATURES ZILNICE
    daily = build_daily_series(df)

    # scor criză
    daily = compute_crisis_score(daily, threshold=crisis_threshold)

    # clustering
    daily, cluster_profile = cluster_daily_regimes(daily)

    # change points
    change_days = detect_change_points(daily)

    # postări critice
    df_critical = extract_critical_posts(df, daily)

    return {
        "df_posts": df,
        "df_daily": daily,
        "df_critical": df_critical,
        "cluster_profile": cluster_profile,
        "change_days": change_days,
        "crisis_days": daily[daily["is_crisis"]],
    }

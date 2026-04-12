"""
pipeline.py
───────────
Full data pipeline:
  1. Fetch raw tables from Supabase
  2. Preprocess + engineer features
  3. Build Genre–Age affinity matrix
  4. Build Genre PII profile
  5. Build Genre × Age risk matrix
  6. Train predictive model
  7. Save every artefact to disk via cache_manager

This module is called ONLY when no cached results exist on disk.
"""

import re
import warnings
import os
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from supabase import create_client
    SUPABASE_OK = True
except ImportError:
    SUPABASE_OK = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                                  classification_report)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_OK = True
except ImportError:
    PRESIDIO_OK = False

try:
    from transformers import pipeline as hf_pipeline
    HF_OK = True
except ImportError:
    HF_OK = False

from cache_manager import save_cache, load_cache

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PII_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION",
    "URL", "IP_ADDRESS", "CRYPTO", "CREDIT_CARD",
    "DATE_TIME", "NRP", "MEDICAL_LICENSE", "US_SSN",
]
SENSITIVITY_MAP = {
    "PERSON": 0.5, "EMAIL_ADDRESS": 0.9, "PHONE_NUMBER": 0.9,
    "LOCATION": 0.6, "URL": 0.3, "IP_ADDRESS": 0.8, "CRYPTO": 0.7,
    "CREDIT_CARD": 1.0, "DATE_TIME": 0.2, "NRP": 0.6,
    "MEDICAL_LICENSE": 0.95, "US_SSN": 1.0,
}
GENRE_RISK_PRIOR = {
    "News & Politics": 0.75, "Education": 0.45, "Entertainment": 0.55,
    "Gaming": 0.50, "Music": 0.40, "Science & Technology": 0.60,
    "Health": 0.70, "Finance": 0.80, "Travel": 0.65, "Food": 0.35,
    "Sports": 0.45, "Fashion & Beauty": 0.50, "Lifestyle": 0.55,
    "Comedy": 0.40, "Other": 0.50,
}
AGE_ORDER = ["18 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65 or older"]
EDU_ORDER = ["High school or equivalent", "Bachelor's degree",
             "Master's degree", "Doctoral degree", "Prefer not to say"]
LIT_ORDER = [
    "Beginner (I know the basics)",
    "Intermediate (I'm comfortable with most technology)",
    "Advanced (I'm very tech-savvy)",
    "Expert (I work in tech or have extensive technical knowledge)",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FETCH FROM SUPABASE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_tables(url: str, key: str, log) -> dict:
    log("Connecting to Supabase…")
    client = create_client(url, key)

    def _get(table, cols="*"):
        try:
            r = client.table(table).select(cols).execute()
            df = pd.DataFrame(r.data)
            log(f"  ✓ {table}: {len(df):,} rows")
            return df
        except Exception as e:
            log(f"  ✗ {table}: {e}")
            return pd.DataFrame()

    survey   = _get("privacy_perception_study")
    comments = _get("comments",
        "comment_id,video_id,comment_text,channel_id,like_count,comment_date")
    videos   = _get("video",
        "video_id,title,view_count,like_count,genre_id,comment_count,channel_id")
    genres   = _get("genre", "genre_id,name")
    channels = _get("channel",
        "channel_id,channel_title,topic_categories,subscriber_count")

    # Merge genre names into videos
    if not genres.empty and not videos.empty:
        genres  = genres.rename(columns={"name": "genre_name"})
        videos  = videos.merge(genres, on="genre_id", how="left")

    return dict(survey=survey, comments=comments,
                videos=videos, genres=genres, channels=channels)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREPROCESS SURVEY
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_survey(df: pd.DataFrame, log) -> pd.DataFrame:
    log("Preprocessing survey…")
    df = df.copy()

    LIKERT = ["q12_concern_data_use", "q13_trust_platforms",
              "q14_perceived_control", "q15_reviews_privacy_settings"]
    for c in LIKERT:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
        out   = df[c].notna() & ~df[c].between(1, 5)
        df.loc[out, c] = np.nan

    # Validity flag
    valid_mask = df[LIKERT].notna().all(axis=1)
    df["is_valid"] = valid_mask

    # Encodings
    df["age_enc"]      = df["q1_age_range"].map(
        {v: i for i, v in enumerate(AGE_ORDER)}).fillna(-1).astype(int)
    df["edu_enc"]      = df["q2_education_level"].map(
        {v: i for i, v in enumerate(EDU_ORDER)}).fillna(-1).astype(int)
    df["literacy_enc"] = df["q3_digital_literacy"].map(
        {v: i for i, v in enumerate(LIT_ORDER)}).fillna(-1).astype(int)

    df["concern_index"] = (
        df["q12_concern_data_use"] +
        (6 - df["q13_trust_platforms"]) +
        (6 - df["q14_perceived_control"])
    ) / 3
    df["behaviour_gap"] = (
        df["q12_concern_data_use"] - df["q15_reviews_privacy_settings"]
    )
    df["has_youtube"] = df.get("q8_has_youtube_account", pd.Series(dtype=str)).apply(
        lambda x: 1 if isinstance(x, str) and x.lower().startswith("yes") else 0)
    df["uses_personalization"] = df.get(
        "q10_uses_youtube_personalization", pd.Series(dtype=str)).apply(
        lambda x: 0 if isinstance(x, str) and ("no" in x.lower() or "don't know" in x.lower())
        else 1)
    df["privacy_gap_bin"] = df.get("q16_privacy_gap", pd.Series(dtype=str)).map(
        {"Yes": 1, "Maybe": 0.5, "No": 0}).fillna(0)
    df["content_types"] = df.get("q9_youtube_content_type", pd.Series(dtype=str)).apply(
        lambda x: [c.strip() for c in str(x).split(",")]
        if pd.notna(x) and str(x) not in ("nan", "") else [])
    df["platforms"] = df.get("q7_social_media_platforms", pd.Series(dtype=str)).apply(
        lambda x: [p.strip() for p in str(x).split(",")]
        if pd.notna(x) and str(x) not in ("nan", "") else [])

    # Risk label from behaviour scenarios
    def _risky(col, vals):
        return df.get(col, pd.Series(dtype=str)).apply(
            lambda x: 1 if isinstance(x, str) and
            any(v.lower() in x.lower() for v in vals) else 0)

    b1 = _risky("q18_app_permissions_scenario",
                 ["grant all permissions without reviewing", "it depends on the app"])
    b2 = _risky("q19_conversation_tracking_scenario",
                 ["assume it's a coincidence", "i wouldn't notice or care"])
    b3 = _risky("q20_youtube_ad_profiling_reaction",
                 ["neutral", "not concerned", "i already knew this and have accepted"])
    b4 = _risky("q21_privacy_policy_email_behaviour",
                 ["ignore it and accept", "i've never received such an email"])
    b5 = df.get("q22_ad_blocker_scenario", pd.Series(dtype=str)).apply(
        lambda x: 1 if isinstance(x, str) and "disable the ad blocker" in x.lower() else 0)

    df["behaviour_score"] = b1 + b2 + b3 + b4 + b5
    df["risk_label"]      = (df["behaviour_score"] >= 3).astype(int)

    log(f"  Survey: {len(df):,} rows, {df['risk_label'].mean()*100:.1f}% high-risk")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREPROCESS COMMENTS  (+ PII extraction if Presidio available)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r"http\S+|www\.\S+", " URL ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _init_pii_engine():
    if not PRESIDIO_OK:
        return None
    try:
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        return AnalyzerEngine(nlp_engine=provider.create_engine())
    except Exception:
        return None


def _extract_pii(engine, text: str) -> list:
    if engine is None or not text:
        return []
    try:
        results = engine.analyze(text=text, entities=PII_ENTITIES,
                                  language="en", score_threshold=0.6)
        return [{"entity_type": r.entity_type, "score": r.score} for r in results]
    except Exception:
        return []


def _sentiment_lexicon(text: str) -> float:
    pos = len(re.findall(
        r"\b(good|great|safe|secure|trusted|love|happy|positive|helpful)\b", text, re.I))
    neg = len(re.findall(
        r"\b(bad|hack|leak|stolen|exposed|tracked|spied|dangerous|risk|scam)\b", text, re.I))
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def preprocess_comments(raw_comments: pd.DataFrame,
                         raw_videos: pd.DataFrame,
                         log) -> pd.DataFrame:
    log("Preprocessing comments…")
    df = raw_comments.copy()
    df["text_clean"] = df.get("comment_text", pd.Series(dtype=str)).apply(_clean_text)
    df = df[df["text_clean"].str.len() > 5].copy()
    df["like_count"] = pd.to_numeric(df.get("like_count", 0), errors="coerce").fillna(0)

    # Merge genre from videos
    if not raw_videos.empty and "genre_name" in raw_videos.columns:
        vid_genre = raw_videos[["video_id", "genre_name"]].drop_duplicates()
        df = df.merge(vid_genre, on="video_id", how="left")
        df["genre_name"] = df["genre_name"].fillna("Unknown")
    else:
        df["genre_name"] = "Unknown"

    log(f"  Comments: {len(df):,} rows, {df['genre_name'].nunique()} genres")

    # PII extraction
    log("  Initialising PII engine (Presidio)…")
    engine = _init_pii_engine()
    if engine:
        log("  Running PII extraction on comments (this takes a while)…")
    else:
        log("  Presidio not available — using heuristic PII features.")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        text     = row.get("text_clean", "")
        pii_hits = _extract_pii(engine, text)
        sens     = [SENSITIVITY_MAP.get(h["entity_type"], 0.3) for h in pii_hits]
        sentiment = _sentiment_lexicon(text)

        records.append({
            "comment_id":          row.get("comment_id", i),
            "pii_count":           len(pii_hits),
            "unique_pii_types":    len({h["entity_type"] for h in pii_hits}),
            "max_sensitivity":     max(sens, default=0.0),
            "mean_sensitivity":    float(np.mean(sens)) if sens else 0.0,
            "raw_risk_score":      float(sum(sens)),
            "mean_pii_confidence": float(np.mean([h["score"] for h in pii_hits]))
                                   if pii_hits else 0.0,
            "pii_density":         len(pii_hits) / max(len(text), 1) * 100,
            "has_financial_pii":   int(any(h["entity_type"] in
                                      {"CREDIT_CARD", "CRYPTO", "US_SSN"} for h in pii_hits)),
            "has_contact_pii":     int(any(h["entity_type"] in
                                      {"EMAIL_ADDRESS", "PHONE_NUMBER"} for h in pii_hits)),
            "has_identity_pii":    int(any(h["entity_type"] in
                                      {"PERSON", "LOCATION", "NRP"} for h in pii_hits)),
            "pii_entity_types":    [h["entity_type"] for h in pii_hits],
            "sentiment_score":     sentiment,
            "text_length":         len(text),
        })

    feat_df = pd.DataFrame(records)
    # Re-attach genre + original columns
    merged = df[["comment_id", "genre_name", "text_clean", "like_count",
                  "video_id"]].reset_index(drop=True).merge(
        feat_df, on="comment_id", how="left")

    log(f"  PII done: {feat_df['pii_count'].sum():,} total hits")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GENRE–AGE AFFINITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_affinity_df(survey_df: pd.DataFrame) -> pd.DataFrame:
    df = survey_df[["id", "age_enc", "q1_age_range",
                     "content_types", "concern_index",
                     "behaviour_gap", "risk_label"]].copy()
    df = df.explode("content_types").rename(columns={"content_types": "genre_label"})
    return df[df["genre_label"].notna() &
              (df["genre_label"] != "nan") &
              (df["genre_label"] != "")]


def build_affinity_matrix(affinity_df: pd.DataFrame) -> pd.DataFrame:
    adf = affinity_df.copy()
    adf["age_label"] = adf["age_enc"].map({i: v for i, v in enumerate(AGE_ORDER)})
    ct = pd.crosstab(adf["genre_label"], adf["age_label"])
    matrix = ct.div(ct.sum(axis=1), axis=0).fillna(0)
    return matrix.reindex(columns=[a for a in AGE_ORDER if a in matrix.columns])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — GENRE PII PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def build_genre_pii_profile(comments_df: pd.DataFrame) -> pd.DataFrame:
    agg = comments_df.groupby("genre_name").agg(
        n_comments         = ("comment_id",       "count"),
        avg_pii_count      = ("pii_count",         "mean"),
        avg_sensitivity    = ("mean_sensitivity",  "mean"),
        avg_risk_score     = ("raw_risk_score",    "mean"),
        pii_rate           = ("pii_count",         lambda x: (x > 0).mean()),
        financial_pii_rate = ("has_financial_pii", "mean"),
        contact_pii_rate   = ("has_contact_pii",   "mean"),
        identity_pii_rate  = ("has_identity_pii",  "mean"),
        avg_sentiment      = ("sentiment_score",   "mean"),
        avg_pii_density    = ("pii_density",       "mean"),
    ).reset_index()

    agg["risk_prior"] = agg["genre_name"].map(GENRE_RISK_PRIOR).fillna(0.5)
    scaler = MinMaxScaler()
    data_sig = scaler.fit_transform(
        agg[["avg_risk_score", "pii_rate", "avg_sensitivity"]]
    ).mean(axis=1)
    agg["genre_risk_index"] = 0.7 * data_sig + 0.3 * agg["risk_prior"]
    return agg.sort_values("genre_risk_index", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — GENRE × AGE RISK MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_matrix(affinity_df: pd.DataFrame,
                       genre_pii_df: pd.DataFrame) -> pd.DataFrame:
    adf = affinity_df.copy()
    adf["age_label"] = adf["age_enc"].map({i: v for i, v in enumerate(AGE_ORDER)})
    gpp = genre_pii_df.set_index("genre_name")

    risk_matrix = {}
    for genre in genre_pii_df["genre_name"].unique():
        genre_risk = gpp.loc[genre, "genre_risk_index"] if genre in gpp.index else 0.5
        row_scores = {}
        for age in adf["age_label"].dropna().unique():
            age_data = adf[adf["age_label"] == age]
            avg_concern = age_data["concern_index"].mean()
            avg_beh_gap = age_data["behaviour_gap"].mean()
            age_enc_val = age_data["age_enc"].iloc[0] if len(age_data) > 0 else 0
            age_factor  = 1.0 + (age_enc_val * 0.03)
            score = (
                0.4 * genre_risk +
                0.25 * (avg_concern / 5.0) +
                0.20 * (max(avg_beh_gap, 0) / 4.0) +
                0.15 * age_factor * 0.5
            )
            row_scores[age] = round(float(np.clip(score, 0, 1)), 4)
        risk_matrix[genre] = row_scores

    matrix = pd.DataFrame(risk_matrix).T
    ordered = [a for a in AGE_ORDER if a in matrix.columns]
    return matrix[ordered]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PREDICTIVE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_model(survey_df: pd.DataFrame,
                affinity_df: pd.DataFrame,
                genre_pii_df: pd.DataFrame,
                log) -> dict:
    if not SKLEARN_OK:
        log("  scikit-learn not available — skipping model.")
        return {}

    log("Training predictive model…")
    gpp = genre_pii_df.set_index("genre_name")

    def _lkp(row, feat, default=0.0):
        g = row.get("genre_label", "Unknown")
        return float(gpp.loc[g, feat]) if g in gpp.index else default

    feat_df = affinity_df.copy()
    feat_df["genre_risk_index"] = feat_df.apply(lambda r: _lkp(r, "genre_risk_index", 0.5), axis=1)
    feat_df["genre_avg_pii"]    = feat_df.apply(lambda r: _lkp(r, "avg_pii_count"), axis=1)
    feat_df["genre_pii_rate"]   = feat_df.apply(lambda r: _lkp(r, "pii_rate"), axis=1)
    feat_df["genre_sentiment"]  = feat_df.apply(lambda r: _lkp(r, "avg_sentiment"), axis=1)

    agg = feat_df.groupby("id").agg(
        genre_risk_index = ("genre_risk_index", "mean"),
        genre_avg_pii    = ("genre_avg_pii",    "mean"),
        genre_pii_rate   = ("genre_pii_rate",   "mean"),
        genre_sentiment  = ("genre_sentiment",  "mean"),
    ).reset_index()

    DEMO_FEAT = [
        "age_enc", "edu_enc", "literacy_enc", "concern_index",
        "behaviour_gap", "has_youtube", "uses_personalization",
        "privacy_gap_bin", "q12_concern_data_use", "q13_trust_platforms",
        "q14_perceived_control", "q15_reviews_privacy_settings",
    ]
    model_df = survey_df[["id"] + DEMO_FEAT + ["risk_label", "is_valid"]].merge(
        agg, on="id", how="left")
    model_df = model_df[model_df["is_valid"]].drop(columns=["is_valid"])
    FEAT_COLS = DEMO_FEAT + ["genre_risk_index", "genre_avg_pii",
                              "genre_pii_rate", "genre_sentiment"]
    avail = [c for c in FEAT_COLS if c in model_df.columns]
    model_df = model_df.dropna(subset=avail + ["risk_label"])

    X = model_df[avail].astype(float)
    y = model_df["risk_label"].astype(int)

    if len(y.unique()) < 2 or len(y) < 30:
        log("  Not enough data for model training.")
        return {}

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.2, stratify=y, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42)

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
    f1s  = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1")

    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    auc_val = roc_auc_score(y_te, y_proba)
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred,
                                                   average="binary", zero_division=0)
    fi = dict(zip(avail, model.feature_importances_.tolist()))
    report = classification_report(y_te, y_pred,
                                    target_names=["Low-Risk", "High-Risk"])

    log(f"  Model AUC-ROC: {auc_val:.4f}  F1: {f:.4f}")
    return {
        "auc_roc":            round(float(auc_val), 4),
        "precision":          round(float(p), 4),
        "recall":             round(float(r), 4),
        "f1":                 round(float(f), 4),
        "cv_auc_mean":        round(float(aucs.mean()), 4),
        "cv_auc_std":         round(float(aucs.std()), 4),
        "cv_f1_mean":         round(float(f1s.mean()), 4),
        "feature_importances": fi,
        "n_train":            int(len(y_tr)),
        "n_test":             int(len(y_te)),
        "class_report":       report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — called by app.py when no cache exists
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(supabase_url: str, supabase_key: str, log) -> bool:
    """
    Runs the full pipeline and writes all results to disk cache.
    Returns True on success, False on failure.
    """
    try:
        log("═" * 55)
        log("  PRIVACY PIPELINE — FIRST RUN (saving to disk cache)")
        log("═" * 55)

        # 1. Fetch
        tables = fetch_all_tables(supabase_url, supabase_key, log)
        save_cache("raw_survey",   tables["survey"],   "Raw survey responses")
        save_cache("raw_comments", tables["comments"], "Raw comments")
        save_cache("raw_videos",   tables["videos"],   "Raw videos + genres")

        # 2. Preprocess survey
        survey_df = preprocess_survey(tables["survey"], log)
        save_cache("survey", survey_df, "Preprocessed survey")

        # 3. Preprocess comments + PII
        comments_df = preprocess_comments(tables["comments"], tables["videos"], log)
        save_cache("comments", comments_df, "Comments with PII features")

        # 4. Genre affinity
        log("Building genre–age affinity…")
        affinity_df  = build_affinity_df(survey_df)
        affinity_mat = build_affinity_matrix(affinity_df)
        save_cache("affinity_df",  affinity_df,  "Affinity long table")
        save_cache("affinity_mat", affinity_mat, "Genre–Age affinity matrix")

        # 5. Genre PII profile
        log("Building genre PII profile…")
        genre_pii = build_genre_pii_profile(comments_df)
        save_cache("genre_pii", genre_pii, "Genre PII profile")

        # 6. Risk matrix
        log("Building risk matrix…")
        risk_matrix = build_risk_matrix(affinity_df, genre_pii)
        save_cache("risk_matrix", risk_matrix, "Genre × Age risk matrix")

        # 7. Model
        model_results = train_model(survey_df, affinity_df, genre_pii, log)
        save_cache("model_results", model_results, "Predictive model metrics")

        log("═" * 55)
        log("  ✓ Pipeline complete — all results saved to disk.")
        log("═" * 55)
        return True

    except Exception as e:
        log(f"✗ Pipeline failed: {e}")
        import traceback
        log(traceback.format_exc())
        return False

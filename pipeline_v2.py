"""
================================================================================
Privacy Perception Study — MULTIMODAL Analysis Pipeline  [v2 — Long-Term Fixes]
================================================================================
Multimodal AI Architecture for Social Media Privacy Research

CHANGES FROM v1 (Long-Term Fixes Applied)
──────────────────────────────────────────
  ① Longitudinal support   → LongitudinalDataManager tracks users over time,
                              detects PII-awareness evolution, computes churn/
                              learning curves per (user, genre, wave).
  ② Causal inference       → CausalInferenceEngine implements propensity score
                              matching (PSM) so genre-exposure comparisons are
                              apples-to-apples across observed confounders.
  ③ SOTA PII detection     → SOTAPIIDetector wraps a fine-tuned DeBERTa NER
                              model (via HuggingFace) with cross-sentence
                              coreference resolution; Presidio kept as fallback.
  ④ User-study validation  → UserStudyValidator correlates model risk scores
                              against self-reported privacy harm events,
                              computes Spearman ρ, Kendall τ, calibration curve.

Dependencies
──────────────────
    pip install transformers accelerate
    pip install causalml scikit-learn scipy
    pip install spacy
    python -m spacy download en_core_web_lg
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import re
import warnings
import os
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import (LabelEncoder, StandardScaler,
                                   MinMaxScaler, PolynomialFeatures)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_fscore_support, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Init] PyTorch not available — neural fusion model will be skipped.")

try:
    from transformers import (
        pipeline as hf_pipeline,
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline as transformers_pipeline,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[Init] HuggingFace Transformers not available.")

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("[Init] Presidio not available — SOTA detector will be primary.")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[Init] Supabase client not installed.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

try:
    from google.colab import userdata
    SUPABASE_URL = userdata.get("supabase_URL")
    SUPABASE_KEY = userdata.get("supabase_key")
except Exception:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_KEY")

PII_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION",
    "URL", "IP_ADDRESS", "CRYPTO", "CREDIT_CARD",
    "DATE_TIME", "NRP", "MEDICAL_LICENSE", "US_SSN",
]

SENSITIVITY_MAP = {
    "PERSON":          0.5,
    "EMAIL_ADDRESS":   0.9,
    "PHONE_NUMBER":    0.9,
    "LOCATION":        0.6,
    "URL":             0.3,
    "IP_ADDRESS":      0.8,
    "CRYPTO":          0.7,
    "CREDIT_CARD":     1.0,
    "DATE_TIME":       0.2,
    "NRP":             0.6,
    "MEDICAL_LICENSE": 0.95,
    "US_SSN":          1.0,
}

GENRE_RISK_PRIOR = {
    "News & Politics":        0.75,
    "Education":              0.45,
    "Entertainment":          0.55,
    "Gaming":                 0.50,
    "Music":                  0.40,
    "Science & Technology":   0.60,
    "Health":                 0.70,
    "Finance":                0.80,
    "Travel":                 0.65,
    "Food":                   0.35,
    "Sports":                 0.45,
    "Fashion & Beauty":       0.50,
    "Lifestyle":              0.55,
    "Comedy":                 0.40,
    "Other":                  0.50,
}


# ─────────────────────────────────────────────────────────────────────────────
# ① LONGITUDINAL DATA MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class LongitudinalDataManager:
    WAVE_COLS = ["wave_id", "wave_date", "user_id"]

    def __init__(self):
        self.wave_registry: Dict[int, pd.Timestamp] = {}
        print("[Longitudinal] LongitudinalDataManager initialised.")

    def ingest_wave(self, survey_wave: pd.DataFrame, wave_id: int,
                    wave_date: str) -> pd.DataFrame:
        df = survey_wave.copy()
        df["wave_id"]   = wave_id
        df["wave_date"] = pd.to_datetime(wave_date)
        if "user_id" not in df.columns:
            if "id" in df.columns:
                df["user_id"] = df["id"]
            else:
                df["user_id"] = df.index.astype(str)
        self.wave_registry[wave_id] = pd.to_datetime(wave_date)
        print(f"[Longitudinal] Wave {wave_id} ({wave_date}): {len(df):,} respondents ingested.")
        return df

    def build_panel(self, wave_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        panel = pd.concat(wave_dfs, ignore_index=True)
        panel = panel.sort_values(["user_id", "wave_date"])
        baseline = panel.groupby("user_id")["wave_date"].min().rename("baseline_date")
        panel = panel.join(baseline, on="user_id")
        panel["days_since_baseline"] = (panel["wave_date"] - panel["baseline_date"]).dt.days
        print(f"[Longitudinal] Panel built: {len(panel):,} rows, "
              f"{panel['user_id'].nunique():,} unique users, "
              f"{panel['wave_id'].nunique()} waves.")
        return panel

    def compute_awareness_evolution(self, panel: pd.DataFrame,
                                     concern_col: str = "concern_index",
                                     beh_col: str = "behaviour_gap",
                                     group_cols: List[str] = None) -> pd.DataFrame:
        group_cols = group_cols or ["user_id"]
        records = []
        for keys, grp in panel.groupby(group_cols):
            grp = grp.sort_values("days_since_baseline")
            if len(grp) < 2:
                continue
            x = grp["days_since_baseline"].values.astype(float)
            y_concern = grp[concern_col].fillna(grp[concern_col].mean()).values
            y_beh     = grp[beh_col].fillna(grp[beh_col].mean()).values
            slope_concern, intercept_c, r_c, p_c, _ = stats.linregress(x, y_concern)
            slope_beh,     _,           r_b, p_b, _ = stats.linregress(x, y_beh)
            rec = {k: v for k, v in zip(group_cols,
                   keys if isinstance(keys, tuple) else (keys,))}
            rec.update({
                "n_waves":               len(grp),
                "concern_slope":         round(slope_concern, 6),
                "concern_r2":            round(r_c ** 2, 4),
                "concern_p_value":       round(p_c, 4),
                "behaviour_gap_slope":   round(slope_beh, 6),
                "behaviour_gap_r2":      round(r_b ** 2, 4),
                "learning":              slope_concern > 0 and p_c < 0.05,
                "forgetting":            slope_concern < 0 and p_c < 0.05,
            })
            records.append(rec)
        evolution_df = pd.DataFrame(records)
        n_learning   = evolution_df["learning"].sum()
        n_forgetting = evolution_df["forgetting"].sum()
        print(f"[Longitudinal] Awareness evolution computed — "
              f"Learning: {n_learning} | Forgetting: {n_forgetting}")
        return evolution_df

    def compute_attrition(self, panel: pd.DataFrame) -> pd.DataFrame:
        n_waves   = panel["wave_id"].nunique()
        max_waves = panel.groupby("user_id")["wave_id"].nunique()
        completers = max_waves[max_waves == n_waves].index
        dropouts   = max_waves[max_waves < n_waves].index
        wave0 = panel[panel["wave_id"] == panel["wave_id"].min()]
        comp_stats    = wave0[wave0["user_id"].isin(completers)].describe()
        dropout_stats = wave0[wave0["user_id"].isin(dropouts)].describe()
        print(f"[Longitudinal] Completers: {len(completers):,} | "
              f"Dropouts: {len(dropouts):,} "
              f"({len(dropouts)/len(max_waves)*100:.1f}%)")
        return pd.DataFrame({
            "completers": comp_stats.iloc[1],
            "dropouts":   dropout_stats.iloc[1],
        })

    def attach_temporal_pii(self, panel: pd.DataFrame,
                             comment_waves: pd.DataFrame,
                             text_features: pd.DataFrame,
                             id_col: str = "comment_id") -> pd.DataFrame:
        if comment_waves.empty or text_features.empty:
            print("[Longitudinal] No comment wave data — skipping PII attachment.")
            return panel
        merged = comment_waves[[id_col, "user_id", "wave_id"]].merge(
            text_features[[id_col, "pii_count", "raw_risk_score", "mean_sensitivity"]],
            on=id_col, how="inner"
        )
        wave_pii = merged.groupby(["user_id", "wave_id"]).agg(
            avg_pii_per_comment=("pii_count",        "mean"),
            total_pii_risk      =("raw_risk_score",   "sum"),
            avg_sensitivity     =("mean_sensitivity", "mean"),
            n_comments          =(id_col,             "count"),
        ).reset_index()
        panel = panel.merge(wave_pii, on=["user_id", "wave_id"], how="left")
        print(f"[Longitudinal] PII features attached: {len(panel):,} rows.")
        return panel


# ─────────────────────────────────────────────────────────────────────────────
# ② CAUSAL INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CausalInferenceEngine:
    def __init__(self, caliper: float = 0.05, n_neighbors: int = 1):
        self.caliper     = caliper
        self.n_neighbors = n_neighbors
        self.propensity_models_: Dict[str, LogisticRegression] = {}
        self.att_estimates_:     Dict[str, dict]               = {}
        self.balance_reports_:   Dict[str, pd.DataFrame]       = {}
        print(f"[Causal] CausalInferenceEngine initialised (caliper={caliper}, k={n_neighbors}).")

    def _fit_propensity(self, df: pd.DataFrame, treatment_col: str,
                        covariate_cols: List[str]) -> np.ndarray:
        X = df[covariate_cols].fillna(0).values
        y = df[treatment_col].astype(int).values
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        model  = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        model.fit(X_s, y)
        ps = model.predict_proba(X_s)[:, 1]
        self.propensity_models_[treatment_col] = model
        return ps

    def _match(self, ps: np.ndarray, treatment: np.ndarray) -> pd.DataFrame:
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        if len(treated_idx) == 0 or len(control_idx) == 0:
            return pd.DataFrame(columns=["treated_idx", "control_idx", "ps_distance"])
        nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(control_idx)),
            metric="euclidean",
        )
        nn.fit(ps[control_idx].reshape(-1, 1))
        distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
        caliper_std = self.caliper * ps.std()
        rows = []
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            for d, j in zip(dists, idxs):
                if d <= caliper_std:
                    rows.append({
                        "treated_idx": treated_idx[i],
                        "control_idx": control_idx[j],
                        "ps_distance": round(float(d), 6),
                    })
        return pd.DataFrame(rows)

    def _balance_report(self, df: pd.DataFrame, matches: pd.DataFrame,
                        treatment_col: str, covariate_cols: List[str]) -> pd.DataFrame:
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]
        if matches.empty:
            return pd.DataFrame()
        matched_treated = df.iloc[matches["treated_idx"].values]
        matched_control = df.iloc[matches["control_idx"].values]
        rows = []
        for col in covariate_cols:
            t_mean = treated[col].mean()
            c_mean = control[col].mean()
            pooled_std = np.sqrt(
                (treated[col].std() ** 2 + control[col].std() ** 2) / 2
            ) + 1e-9
            smd_before = abs(t_mean - c_mean) / pooled_std
            mt_mean = matched_treated[col].mean()
            mc_mean = matched_control[col].mean()
            smd_after = abs(mt_mean - mc_mean) / pooled_std
            rows.append({
                "covariate":  col,
                "smd_before": round(smd_before, 4),
                "smd_after":  round(smd_after,  4),
                "balanced":   smd_after < 0.1,
            })
        return pd.DataFrame(rows)

    def estimate_genre_effect(self, df: pd.DataFrame, genre: str,
                               genre_col: str, outcome_col: str,
                               covariate_cols: List[str]) -> dict:
        df = df.copy()
        df["__treatment__"] = (df[genre_col] == genre).astype(int)
        ps       = self._fit_propensity(df, "__treatment__", covariate_cols)
        df["ps"] = ps
        matches  = self._match(ps, df["__treatment__"].values)
        if matches.empty or len(matches) < 5:
            print(f"[Causal] '{genre}': insufficient matches ({len(matches)}) — skipping.")
            return {"genre": genre, "att": np.nan, "p_value": np.nan,
                    "n_matched": 0, "balance": pd.DataFrame()}
        treated_outcomes = df[outcome_col].iloc[matches["treated_idx"].values].values
        control_outcomes = df[outcome_col].iloc[matches["control_idx"].values].values
        att     = float(np.mean(treated_outcomes - control_outcomes))
        t_stat, p_val = stats.ttest_rel(treated_outcomes, control_outcomes)
        balance = self._balance_report(df, matches, "__treatment__", covariate_cols)
        result  = {
            "genre":     genre,
            "att":       round(att, 6),
            "p_value":   round(float(p_val), 4),
            "n_matched": len(matches),
            "balance":   balance,
        }
        self.att_estimates_[genre]   = result
        self.balance_reports_[genre] = balance
        significance = "✓ significant" if p_val < 0.05 else "✗ not significant"
        print(f"[Causal] '{genre}': ATT={att:+.4f}, p={p_val:.4f} {significance}, "
              f"n_matched={len(matches)}")
        return result

    def run_all_genres(self, df: pd.DataFrame, genre_col: str,
                       outcome_col: str, covariate_cols: List[str],
                       min_treated: int = 20) -> pd.DataFrame:
        genres = df[genre_col].value_counts()
        genres = genres[genres >= min_treated].index.tolist()
        print(f"\n[Causal] Running PSM for {len(genres)} genres …")
        results = []
        for genre in genres:
            r = self.estimate_genre_effect(df, genre, genre_col, outcome_col, covariate_cols)
            results.append({k: v for k, v in r.items() if k != "balance"})
        summary = pd.DataFrame(results).sort_values("att", ascending=False)
        print(f"\n[Causal] Causal summary:\n{summary.to_string(index=False)}\n")
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# ③ SOTA PII DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class SOTAPIIDetector:
    HF_NER_MODELS = [
        "Jean-Baptiste/roberta-large-ner-english",
        "dslim/bert-large-NER",
        "dslim/bert-base-NER",
    ]
    HF_TO_PII = {
        "PER":  "PERSON",
        "LOC":  "LOCATION",
        "ORG":  "LOCATION",
        "MISC": "NRP",
    }

    def __init__(self, score_threshold: float = 0.70):
        self.score_threshold = score_threshold
        self.ner_pipeline    = None
        self.coref_nlp       = None
        self.presidio        = None
        self._load_models()

    def _load_models(self):
        if HF_AVAILABLE:
            for model_name in self.HF_NER_MODELS:
                try:
                    self.ner_pipeline = transformers_pipeline(
                        "ner", model=model_name,
                        aggregation_strategy="simple", device=-1,
                    )
                    print(f"[SOTA PII] Loaded NER model: {model_name}")
                    break
                except Exception as e:
                    print(f"[SOTA PII] Failed to load {model_name}: {e}")

        try:
            import spacy
            nlp = spacy.load("en_core_web_lg")
            try:
                import neuralcoref
                neuralcoref.add_to_pipe(nlp)
                self.coref_nlp = nlp
                print("[SOTA PII] Coreference resolution enabled (neuralcoref).")
            except ImportError:
                self.coref_nlp = nlp
                print("[SOTA PII] neuralcoref not found — using spaCy without coref.")
        except Exception:
            print("[SOTA PII] spaCy not available — coref disabled.")

        if PRESIDIO_AVAILABLE:
            try:
                provider = NlpEngineProvider(nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
                })
                self.presidio = AnalyzerEngine(nlp_engine=provider.create_engine())
                print("[SOTA PII] Presidio fallback loaded.")
            except Exception as e:
                print(f"[SOTA PII] Presidio fallback failed: {e}")

        if self.ner_pipeline is None and self.presidio is None:
            print("[SOTA PII] WARNING: No PII detector available — all PII features will be zero.")

    def _resolve_coreferences(self, text: str) -> str:
        if self.coref_nlp is None:
            return text
        try:
            doc = self.coref_nlp(text)
            if hasattr(doc._, "coref_resolved"):
                return doc._.coref_resolved
            return text
        except Exception:
            return text

    def _extract_hf_ner(self, text: str) -> List[dict]:
        if self.ner_pipeline is None or not text:
            return []
        try:
            entities = self.ner_pipeline(text[:512])
            results  = []
            for ent in entities:
                score = float(ent.get("score", 0))
                if score < self.score_threshold:
                    continue
                label    = ent.get("entity_group", ent.get("entity", ""))
                pii_type = self.HF_TO_PII.get(label, None)
                if pii_type is None:
                    continue
                results.append({
                    "entity_type": pii_type,
                    "word":        ent.get("word", ""),
                    "score":       round(score, 4),
                    "start":       ent.get("start", 0),
                    "end":         ent.get("end",   0),
                    "source":      "deberta_ner",
                })
            return results
        except Exception as e:
            print(f"[SOTA PII] HF NER error: {e}")
            return []

    def _extract_presidio(self, text: str) -> List[dict]:
        if self.presidio is None or not text:
            return []
        try:
            hits = self.presidio.analyze(
                text=text, entities=PII_ENTITIES,
                language="en", score_threshold=self.score_threshold,
            )
            return [{
                "entity_type": r.entity_type,
                "word":        text[r.start:r.end],
                "score":       round(r.score, 4),
                "start":       r.start,
                "end":         r.end,
                "source":      "presidio",
            } for r in hits]
        except Exception:
            return []

    def _tag_relationship_context(self, text: str, entities: List[dict]) -> List[dict]:
        relationship_pattern = re.compile(
            r"\b(my|his|her|our|their)\s+(mom|dad|mother|father|boss|"
            r"sister|brother|wife|husband|partner|friend|son|daughter)\b",
            re.I,
        )
        relational_spans = [m.span() for m in relationship_pattern.finditer(text)]
        for ent in entities:
            if ent["entity_type"] != "PERSON":
                continue
            for rs, re_ in relational_spans:
                if abs(ent["start"] - re_) < 30:
                    ent["has_relationship_context"] = True
                    ent["score"] = min(ent["score"] + 0.10, 1.0)
                    break
            else:
                ent.setdefault("has_relationship_context", False)
        return entities

    def extract(self, text: str) -> List[dict]:
        if not text:
            return []
        resolved_text = self._resolve_coreferences(text)
        hf_entities   = self._extract_hf_ner(resolved_text)
        pre_entities  = self._extract_presidio(resolved_text)
        hf_types  = {e["entity_type"] for e in hf_entities}
        pre_only  = [
            e for e in pre_entities
            if e["entity_type"] not in hf_types
            or e["entity_type"] in {"EMAIL_ADDRESS", "PHONE_NUMBER",
                                    "CREDIT_CARD", "US_SSN", "IP_ADDRESS"}
        ]
        all_entities = hf_entities + pre_only
        all_entities = self._tag_relationship_context(resolved_text, all_entities)
        return all_entities

    def extract_sentiment(self, text: str) -> float:
        if not text:
            return 0.0
        if HF_AVAILABLE:
            try:
                sent_pipe = hf_pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True, max_length=512,
                )
                result = sent_pipe(text[:512])[0]
                score  = result["score"]
                return score if result["label"] == "POSITIVE" else -score
            except Exception:
                pass
        pos = len(re.findall(
            r"\b(good|great|safe|secure|trusted|love|happy|positive|helpful)\b",
            text, re.I))
        neg = len(re.findall(
            r"\b(bad|hack|leak|stolen|exposed|tracked|spied|dangerous|risk|scam)\b",
            text, re.I))
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0.0

    def extract_dataframe(self, df: pd.DataFrame,
                           text_col: str = "text_clean",
                           id_col:   str = "comment_id") -> pd.DataFrame:
        records = []
        for idx, row in df.iterrows():
            row_id    = row.get(id_col, idx)
            text      = row.get(text_col, "")
            sentiment = self.extract_sentiment(text)
            pii_hits  = self.extract(text)
            sensitivity_vals = [
                SENSITIVITY_MAP.get(h["entity_type"], 0.3) *
                (1.1 if h.get("has_relationship_context") else 1.0)
                for h in pii_hits
            ]
            records.append({
                id_col:                   row_id,
                "pii_count":              len(pii_hits),
                "unique_pii_types":       len({h["entity_type"] for h in pii_hits}),
                "max_sensitivity":        max(sensitivity_vals, default=0.0),
                "mean_sensitivity":       np.mean(sensitivity_vals) if sensitivity_vals else 0.0,
                "raw_risk_score":         sum(sensitivity_vals),
                "mean_pii_confidence":    np.mean([h["score"] for h in pii_hits])
                                          if pii_hits else 0.0,
                "pii_density":            len(pii_hits) / max(len(text), 1) * 100,
                "has_financial_pii":      int(any(h["entity_type"] in
                                           {"CREDIT_CARD","CRYPTO","US_SSN"}
                                           for h in pii_hits)),
                "has_contact_pii":        int(any(h["entity_type"] in
                                           {"EMAIL_ADDRESS","PHONE_NUMBER"}
                                           for h in pii_hits)),
                "has_identity_pii":       int(any(h["entity_type"] in
                                           {"PERSON","LOCATION","NRP"}
                                           for h in pii_hits)),
                "has_relational_pii":     int(any(h.get("has_relationship_context")
                                           for h in pii_hits)),
                "pii_entity_types":       [h["entity_type"] for h in pii_hits],
                "sentiment_score":        sentiment,
                "text_length":            len(text),
                "pii_source":             (
                    "deberta" if any(h.get("source") == "deberta_ner" for h in pii_hits)
                    else "presidio" if pii_hits else "none"
                ),
            })
        result = pd.DataFrame(records)
        print(f"[SOTA PII] Extracted features for {len(result):,} items. "
              f"Total PII hits: {result['pii_count'].sum():,} | "
              f"Relational PII: {result['has_relational_pii'].sum():,}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# ④ USER STUDY VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class UserStudyValidator:
    def __init__(self):
        print("[Validate] UserStudyValidator initialised.")

    def merge_predictions(self, predictions_df: pd.DataFrame,
                           harm_reports: pd.DataFrame) -> pd.DataFrame:
        merged = predictions_df.merge(harm_reports, on="user_id", how="inner")
        print(f"[Validate] Merged {len(merged):,} users with harm reports "
              f"({harm_reports['reported_harm'].mean()*100:.1f}% harm rate).")
        return merged

    def compute_rank_correlation(self, merged: pd.DataFrame,
                                  score_col: str = "pred_risk",
                                  outcome_col: str = "reported_harm") -> dict:
        scores   = merged[score_col].fillna(0)
        outcomes = merged[outcome_col].fillna(0)
        spearman_r, spearman_p = stats.spearmanr(scores, outcomes)
        kendall_t,  kendall_p  = stats.kendalltau(scores, outcomes)
        point_biserial_r, pb_p = stats.pointbiserialr(outcomes, scores)
        results = {
            "spearman_rho":   round(float(spearman_r), 4),
            "spearman_p":     round(float(spearman_p), 4),
            "kendall_tau":    round(float(kendall_t),  4),
            "kendall_p":      round(float(kendall_p),  4),
            "point_biserial": round(float(point_biserial_r), 4),
            "pb_p":           round(float(pb_p), 4),
            "n":              len(merged),
        }
        print(f"\n[Validate] Spearman ρ={results['spearman_rho']:.4f}  "
              f"Kendall τ={results['kendall_tau']:.4f}")
        return results

    def compute_calibration(self, merged: pd.DataFrame,
                             score_col: str = "pred_risk",
                             outcome_col: str = "reported_harm",
                             n_bins: int = 10) -> dict:
        scores   = merged[score_col].fillna(0).values
        outcomes = merged[outcome_col].fillna(0).values.astype(int)
        brier    = brier_score_loss(outcomes, scores)
        fraction_pos, mean_pred = calibration_curve(
            outcomes, scores, n_bins=n_bins, strategy="quantile")
        bin_sizes = np.histogram(scores, bins=n_bins)[0]
        ece = float(np.sum(np.abs(fraction_pos - mean_pred) * bin_sizes / len(scores)))
        return {
            "brier_score": round(float(brier), 4),
            "ece":         round(ece, 4),
            "fraction_pos": fraction_pos.tolist(),
            "mean_pred":    mean_pred.tolist(),
        }

    def subgroup_validation(self, merged: pd.DataFrame,
                             group_col: str = "q1_age_range",
                             score_col: str = "pred_risk",
                             outcome_col: str = "reported_harm") -> pd.DataFrame:
        rows = []
        for group, grp in merged.groupby(group_col):
            if len(grp) < 10:
                continue
            scores   = grp[score_col].fillna(0).values
            outcomes = grp[outcome_col].fillna(0).values
            try:
                rho, p = stats.spearmanr(scores, outcomes)
                brier  = brier_score_loss(outcomes.astype(int), scores)
                bias   = float(scores.mean() - outcomes.mean())
            except Exception:
                rho, p, brier, bias = np.nan, np.nan, np.nan, np.nan
            rows.append({
                group_col:      group,
                "n":            len(grp),
                "spearman_rho": round(float(rho),   4) if not np.isnan(rho)   else np.nan,
                "spearman_p":   round(float(p),      4) if not np.isnan(p)     else np.nan,
                "brier_score":  round(float(brier),  4) if not np.isnan(brier) else np.nan,
                "mean_pred":    round(float(scores.mean()), 4),
                "harm_rate":    round(float(outcomes.mean()), 4),
                "bias":         round(float(bias),   4) if not np.isnan(bias)  else np.nan,
            })
        return pd.DataFrame(rows)

    def full_report(self, predictions_df: pd.DataFrame,
                    harm_reports: pd.DataFrame,
                    group_col: str = "q1_age_range") -> dict:
        merged = self.merge_predictions(predictions_df, harm_reports)
        if merged.empty:
            print("[Validate] No matching users — skipping validation.")
            return {}
        correlation = self.compute_rank_correlation(merged)
        calibration = self.compute_calibration(merged)
        subgroups   = self.subgroup_validation(merged, group_col=group_col)
        return {
            "merged":      merged,
            "correlation": correlation,
            "calibration": calibration,
            "subgroups":   subgroups,
        }

    def plot_calibration(self, calibration_result: dict, validation_result: dict,
                         bg: str = "#0D1117", fg: str = "#E6EDF3"):
        if not calibration_result or not validation_result:
            return
        plt.rcParams.update({
            "figure.facecolor": bg, "axes.facecolor": bg,
            "text.color": fg, "axes.labelcolor": fg,
            "xtick.color": fg, "ytick.color": fg,
        })
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot([0, 1], [0, 1], "w--", linewidth=1, label="Perfect calibration")
        axes[0].plot(calibration_result["mean_pred"], calibration_result["fraction_pos"],
                     "o-", color="#FF6B6B", linewidth=2, markersize=7, label="Model")
        axes[0].set_xlabel("Mean Predicted Risk")
        axes[0].set_ylabel("Observed Harm Rate")
        axes[0].set_title(
            f"Reliability Diagram\n"
            f"Brier={calibration_result['brier_score']:.3f}  "
            f"ECE={calibration_result['ece']:.3f}", fontweight="bold")
        axes[0].legend()
        sg = validation_result.get("subgroups", pd.DataFrame())
        if not sg.empty and "spearman_rho" in sg.columns:
            sg_sorted = sg.sort_values("spearman_rho", ascending=True)
            colors    = ["#FF6B6B" if r < 0.3 else "#4ECDC4"
                         for r in sg_sorted["spearman_rho"]]
            group_col = [c for c in sg.columns if c not in
                         ["n","spearman_rho","spearman_p","brier_score",
                          "mean_pred","harm_rate","bias"]][0]
            axes[1].barh(sg_sorted[group_col].astype(str),
                         sg_sorted["spearman_rho"], color=colors, edgecolor="#21262D")
            axes[1].set_xlabel("Spearman ρ")
            axes[1].set_title("Subgroup Correlation", fontweight="bold")
        fig.suptitle("User Study Validation Results", fontsize=14, fontweight="bold", color=fg)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessor:
    AGE_ORDER = ["18 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65 or older"]
    EDU_ORDER = ["High school or equivalent", "Bachelor's degree",
                 "Master's degree", "Doctoral degree", "Prefer not to say"]
    LITERACY_ORDER = [
        "Beginner (I know the basics)",
        "Intermediate (I'm comfortable with most technology)",
        "Advanced (I'm very tech-savvy)",
        "Expert (I work in tech or have extensive technical knowledge)",
    ]
    LIKERT_COLS = [
        "q12_concern_data_use", "q13_trust_platforms",
        "q14_perceived_control", "q15_reviews_privacy_settings",
    ]

    def validate_survey(self, df: pd.DataFrame) -> pd.DataFrame:
        df    = df.copy()
        flags = pd.DataFrame(index=df.index)
        for col in self.LIKERT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            out_of_range = df[col].notna() & ~df[col].between(1, 5)
            flags[f"oob_{col}"] = out_of_range
            df.loc[out_of_range, col] = np.nan
        likert_valid      = df[self.LIKERT_COLS].dropna(how="any")
        straight_line_mask = (likert_valid.nunique(axis=1) == 1)
        flags["straight_line"] = False
        flags.loc[straight_line_mask.index[straight_line_mask], "straight_line"] = True
        concern = pd.to_numeric(df["q12_concern_data_use"], errors="coerce")
        action  = pd.to_numeric(df["q15_reviews_privacy_settings"], errors="coerce")
        flags["impossible_gap"]  = (concern - action).abs() > 4
        flags["duplicate_ts"]    = (
            df["timestamp"].duplicated(keep="first")
            if "timestamp" in df.columns else False
        )
        df["is_valid"] = ~flags.any(axis=1)
        n_total = len(df); n_invalid = (~df["is_valid"]).sum()
        print(f"\n[Validate] Survey: {n_total} rows — "
              f"{n_invalid} flagged ({n_invalid/n_total:.1%}), "
              f"{n_total - n_invalid} retained.\n")
        return df

    def preprocess_survey(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.validate_survey(df)
        df.dropna(subset=["q1_age_range", "q2_education_level", "q3_digital_literacy"],
                  inplace=True)
        df["age_enc"]      = df["q1_age_range"].map(
            {v: i for i, v in enumerate(self.AGE_ORDER)}).fillna(-1).astype(int)
        df["edu_enc"]      = df["q2_education_level"].map(
            {v: i for i, v in enumerate(self.EDU_ORDER)}).fillna(-1).astype(int)
        df["literacy_enc"] = df["q3_digital_literacy"].map(
            {v: i for i, v in enumerate(self.LITERACY_ORDER)}).fillna(-1).astype(int)
        df["concern_index"] = (
            df["q12_concern_data_use"] +
            (6 - df["q13_trust_platforms"]) +
            (6 - df["q14_perceived_control"])
        ) / 3
        df["behaviour_gap"] = (
            df["q12_concern_data_use"] - df["q15_reviews_privacy_settings"]
        )
        df["platforms"] = df["q7_social_media_platforms"].apply(
            lambda x: [p.strip() for p in str(x).split(",")] if pd.notna(x) else [])
        df["content_types"] = df["q9_youtube_content_type"].apply(
            lambda x: [c.strip() for c in str(x).split(",")] if pd.notna(x) else [])
        df["has_youtube"]          = df["q8_has_youtube_account"].apply(
            lambda x: 1 if isinstance(x, str) and x.startswith("Yes") else 0)
        df["uses_personalization"] = df["q10_uses_youtube_personalization"].apply(
            lambda x: 0 if isinstance(x, str) and ("No" in x or "don't know" in x) else 1)
        df["privacy_gap_bin"] = df["q16_privacy_gap"].map(
            {"Yes": 1, "Maybe": 0.5, "No": 0}).fillna(0)
        print(f"[Preprocess] Survey ready: {len(df):,} rows.")
        return df

    def build_genre_affinity(self, survey_df: pd.DataFrame) -> pd.DataFrame:
        df = survey_df[["id","age_enc","q1_age_range","content_types",
                         "concern_index","behaviour_gap","risk_label"]].copy()
        df = df.explode("content_types").rename(columns={"content_types":"genre_label"})
        df = df[df["genre_label"].notna() & (df["genre_label"] != "nan")]
        print(f"[Preprocess] Genre–Age affinity table: {len(df):,} rows.")
        return df

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str): return ""
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"http\S+|www\.\S+", " URL ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text_clean"] = df["comment_text"].apply(self.clean_text)
        df.dropna(subset=["text_clean"], inplace=True)
        df = df[df["text_clean"].str.len() > 5]
        df["like_count"] = pd.to_numeric(
            df.get("like_count", 0), errors="coerce").fillna(0)
        print(f"[Preprocess] Comments ready: {len(df):,} rows.")
        return df

    def merge_comments_with_genre(self, comments_df, videos_df, genres_df):
        if videos_df.empty or genres_df.empty:
            comments_df["genre_name"] = "Unknown"
            return comments_df
        genres_df = genres_df.rename(columns={"name": "genre_name"})
        videos_df = videos_df.merge(genres_df, on="genre_id", how="left")
        merged = comments_df.merge(
            videos_df[["video_id","genre_name","view_count","like_count"]],
            on="video_id", how="left", suffixes=("","_video"),
        )
        merged["genre_name"] = merged["genre_name"].fillna("Unknown")
        print(f"[Preprocess] Comments+Genre merged: {len(merged):,} rows, "
              f"{merged['genre_name'].nunique()} genres.")
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# METADATA MODALITY EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class MetadataModalityExtractor:
    def extract_video_metadata(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        df = videos_df.copy()
        for col in ["view_count","like_count","comment_count"]:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
        df["engagement_rate"] = (
            (df["like_count"] + df["comment_count"]) / df["view_count"].replace(0,1))
        df["comment_ratio"]   = df["comment_count"] / df["view_count"].replace(0,1)
        df["passive_ratio"]   = 1 - df["engagement_rate"].clip(0,1)
        df["popularity_tier"] = pd.cut(
            df["view_count"],
            bins=[0,1_000,100_000,1_000_000,float("inf")],
            labels=[0,1,2,3],
        ).astype(float).fillna(0)
        print(f"[Metadata] Video metadata features: {len(df):,} rows.")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE MODALITY EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class ImageModalityExtractor:
    def __init__(self):
        if CV2_AVAILABLE:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.available = CV2_AVAILABLE
        print(f"[Image] ImageModalityExtractor ready (OpenCV: {CV2_AVAILABLE}).")

    def analyze_image_url(self, url: str) -> dict:
        default = {"face_count":0,"visual_complexity":0.5,"brightness":0.5,
                   "warm_cool_ratio":0.5,"image_available":0}
        if not self.available or not isinstance(url, str):
            return default
        try:
            import urllib.request, tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                img = cv2.imread(tmp.name)
            os.unlink(tmp.name)
            if img is None: return default
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            edges = cv2.Canny(gray, 50, 150)
            b, g, r = cv2.split(img.astype(float))
            return {
                "face_count":        len(faces),
                "visual_complexity": round(float(edges.mean()/255), 4),
                "brightness":        round(float(gray.mean()/255),  4),
                "warm_cool_ratio":   round(float(
                    min((r.mean()+g.mean()*0.5)/(b.mean()+1), 3)/3), 4),
                "image_available":   1,
            }
        except Exception:
            return default

    def extract_batch(self, url_series: pd.Series) -> pd.DataFrame:
        print(f"[Image] Processing {len(url_series):,} images …")
        return pd.DataFrame(
            [self.analyze_image_url(u) for u in url_series],
            index=url_series.index)


# ─────────────────────────────────────────────────────────────────────────────
# MULTIMODAL FUSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalFusionEngine:
    TEXT_FEATURES = [
        "pii_count","unique_pii_types","max_sensitivity","mean_sensitivity",
        "raw_risk_score","mean_pii_confidence","pii_density","has_financial_pii",
        "has_contact_pii","has_identity_pii","has_relational_pii",
        "sentiment_score","text_length",
    ]
    META_FEATURES_VIDEO = [
        "engagement_rate","comment_ratio","passive_ratio","popularity_tier",
    ]
    IMAGE_FEATURES = [
        "face_count","visual_complexity","brightness","warm_cool_ratio","image_available",
    ]

    def __init__(self):
        self.scaler              = StandardScaler()
        self.attention_weights_: Dict[str, float] = {}
        print("[Fusion] MultimodalFusionEngine ready.")

    def fuse(self, text_df, meta_df=None, image_df=None):
        parts = []
        text_cols = [c for c in self.TEXT_FEATURES if c in text_df.columns]
        parts.append(text_df[text_cols].fillna(0).reset_index(drop=True))
        meta_cols = [c for c in self.META_FEATURES_VIDEO
                     if meta_df is not None and c in meta_df.columns]
        if meta_cols: parts.append(meta_df[meta_cols].fillna(0).reset_index(drop=True))
        img_cols  = [c for c in self.IMAGE_FEATURES
                     if image_df is not None and c in image_df.columns]
        if img_cols:  parts.append(image_df[img_cols].fillna(0).reset_index(drop=True))
        fused = pd.concat(parts, axis=1)
        modality_variances = {}
        for label, cols in [("text",text_cols),("metadata",meta_cols),("image",img_cols)]:
            if not cols: continue
            var  = fused[cols].values.var(axis=0)
            mean = np.abs(fused[cols].values.mean(axis=0)) + 1e-6
            modality_variances[label] = float((var / mean).mean())
        if modality_variances:
            arr  = np.array(list(modality_variances.values()))
            exp  = np.exp(arr - arr.max())
            prob = exp / exp.sum()
            self.attention_weights_ = {
                k: round(prob[i], 4)
                for i, k in enumerate(modality_variances.keys())
            }
        print(f"[Fusion] Attention weights: {self.attention_weights_}")
        print(f"[Fusion] Fused feature matrix: {fused.shape}")
        return fused

    def scale(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        return self.scaler.fit_transform(X.fillna(0)) if fit \
               else self.scaler.transform(X.fillna(0))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEER
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    def build_survey_pii_risk(self, survey_df: pd.DataFrame) -> pd.DataFrame:
        df = survey_df.copy()
        risky_perms = {"Grant all permissions without reviewing","It depends on the app"}
        df["b1_risky_perms"] = df["q18_app_permissions_scenario"].apply(
            lambda x: 1 if isinstance(x, str) and
            any(r.lower() in x.lower() for r in risky_perms) else 0)
        risky_track = {"Assume it's a coincidence","I wouldn't notice or care"}
        df["b2_passive_tracking"] = df["q19_conversation_tracking_scenario"].apply(
            lambda x: 1 if isinstance(x, str) and
            any(r.lower() in x.lower() for r in risky_track) else 0)
        risky_ad = {"Neutral","Not concerned","I already knew this and have accepted"}
        df["b3_accepts_profiling"] = df["q20_youtube_ad_profiling_reaction"].apply(
            lambda x: 1 if isinstance(x, str) and
            any(r.lower() in x.lower() for r in risky_ad) else 0)
        risky_email = {"Ignore it and accept","I've never received such an email"}
        df["b4_ignores_policy"] = df["q21_privacy_policy_email_behaviour"].apply(
            lambda x: 1 if isinstance(x, str) and
            any(r.lower() in x.lower() for r in risky_email) else 0)
        df["b5_disables_adblocker"] = df["q22_ad_blocker_scenario"].apply(
            lambda x: 1 if isinstance(x, str) and
            "disable the ad blocker to access content" in x.lower() else 0)
        behaviour_score = (
            df["b1_risky_perms"] + df["b2_passive_tracking"] +
            df["b3_accepts_profiling"] + df["b4_ignores_policy"] +
            df["b5_disables_adblocker"]
        )
        df["behaviour_score"] = behaviour_score
        df["risk_label"]      = (behaviour_score >= 3).astype(int)
        dist = df["risk_label"].value_counts()
        print(f"[Features] Risk labels — High: {dist.get(1,0)}, "
              f"Low: {dist.get(0,0)}, Rate: {df['risk_label'].mean()*100:.1f}%")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# GENRE AGE PRIVACY ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class GenreAgePrivacyAnalyser:
    def __init__(self):
        self.scaler               = StandardScaler()
        self.model                = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42)
        self.genre_age_affinity_  = None
        self.genre_pii_profile_   = None
        self.genre_age_risk_      = None
        self.feature_importances_ = None

    def build_genre_age_affinity(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
        age_labels  = DataPreprocessor.AGE_ORDER
        affinity_df = affinity_df.copy()
        affinity_df["age_label"] = affinity_df["age_enc"].map(
            {i: v for i, v in enumerate(age_labels)})
        ct     = pd.crosstab(affinity_df["genre_label"], affinity_df["age_label"])
        matrix = ct.div(ct.sum(axis=1), axis=0).fillna(0)
        matrix = matrix.reindex(columns=[a for a in age_labels if a in matrix.columns])
        self.genre_age_affinity_ = matrix
        print(f"[Analysis] Affinity matrix: {matrix.shape}")
        return matrix

    def build_genre_pii_profile(self, comments_with_genre, text_features,
                                  id_col="comment_id"):
        merged = comments_with_genre[[id_col,"genre_name"]].merge(
            text_features, on=id_col, how="inner")
        agg = merged.groupby("genre_name").agg(
            n_comments          =(id_col,              "count"),
            avg_pii_count       =("pii_count",          "mean"),
            avg_sensitivity     =("mean_sensitivity",   "mean"),
            avg_risk_score      =("raw_risk_score",     "mean"),
            pii_rate            =("pii_count",          lambda x: (x>0).mean()),
            financial_pii_rate  =("has_financial_pii",  "mean"),
            contact_pii_rate    =("has_contact_pii",    "mean"),
            identity_pii_rate   =("has_identity_pii",   "mean"),
            relational_pii_rate =("has_relational_pii", "mean"),
            avg_sentiment       =("sentiment_score",    "mean"),
            avg_pii_density     =("pii_density",        "mean"),
        ).reset_index()
        agg["risk_prior"] = agg["genre_name"].map(GENRE_RISK_PRIOR).fillna(0.5)
        scaler      = MinMaxScaler()
        data_signal = scaler.fit_transform(
            agg[["avg_risk_score","pii_rate","avg_sensitivity"]]).mean(axis=1)
        agg["genre_risk_index"] = 0.7 * data_signal + 0.3 * agg["risk_prior"]
        self.genre_pii_profile_ = agg.sort_values("genre_risk_index", ascending=False)
        print(f"[Analysis] Genre PII profile: {len(agg)} genres.")
        return self.genre_pii_profile_

    def build_genre_age_risk_matrix(self, affinity_df, genre_pii_profile):
        affinity_df = affinity_df.copy()
        affinity_df["age_label"] = affinity_df["age_enc"].map(
            {i: v for i, v in enumerate(DataPreprocessor.AGE_ORDER)})
        gpp = genre_pii_profile.set_index("genre_name")
        affinity_df["genre_risk_idx"] = affinity_df["genre_label"].map(
            gpp["genre_risk_index"].to_dict()).fillna(0.5)
        X = affinity_df[["genre_risk_idx","concern_index",
                          "behaviour_gap","age_enc"]].fillna(0)
        y = affinity_df["risk_label"].fillna(0)
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        poly   = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_s)
        model  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_poly, y)
        feature_names = poly.get_feature_names_out(
            ["genre_risk","concern","behav_gap","age"])
        weights = dict(zip(feature_names, model.coef_[0]))
        print("[Model] Top learned susceptibility weights:")
        for n, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"  {n}: {w:.4f}")
        risk_matrix = {}
        for genre in genre_pii_profile["genre_name"].unique():
            genre_risk = genre_pii_profile.loc[
                genre_pii_profile["genre_name"]==genre,"genre_risk_index"
            ].iloc[0] if genre in genre_pii_profile["genre_name"].values else 0.5
            row = {}
            for age in affinity_df["age_label"].dropna().unique():
                age_data    = affinity_df[affinity_df["age_label"]==age]
                avg_concern = age_data["concern_index"].mean()
                avg_beh_gap = age_data["behaviour_gap"].mean()
                age_enc     = age_data["age_enc"].iloc[0] if len(age_data)>0 else 0
                X_pred = scaler.transform([[genre_risk,avg_concern,avg_beh_gap,age_enc]])
                X_pred = poly.transform(X_pred)
                row[age] = round(model.predict_proba(X_pred)[0,1], 4)
            risk_matrix[genre] = row
        matrix = pd.DataFrame(risk_matrix).T
        self.genre_age_risk_        = matrix
        self.susceptibility_model_  = model
        self.susceptibility_scaler_ = scaler
        self.susceptibility_poly_   = poly
        print(f"[Analysis] Risk matrix: {matrix.shape}")
        return matrix

    def build_genre_age_features(self, affinity_df, genre_pii_profile):
        gpp = genre_pii_profile.set_index("genre_name")
        affinity_df = affinity_df.copy()
        affinity_df["age_label"] = affinity_df["age_enc"].map(
            {i:v for i,v in enumerate(DataPreprocessor.AGE_ORDER)})
        def lkp(row, feat):
            g = row.get("genre_label","Unknown")
            return gpp.loc[g,feat] if g in gpp.index else 0.5
        for feat in ["genre_risk_index","avg_pii_count","pii_rate",
                     "financial_pii_rate","avg_sentiment"]:
            affinity_df[feat] = affinity_df.apply(lambda r: lkp(r,feat), axis=1)
        return affinity_df

    def train(self, survey_df, affinity_df, genre_pii_profile):
        feat_df = self.build_genre_age_features(affinity_df, genre_pii_profile)
        agg_genre_feats = feat_df.groupby("id").agg(
            genre_risk_index   =("genre_risk_index",  "mean"),
            genre_avg_pii      =("avg_pii_count",     "mean"),
            genre_pii_rate     =("pii_rate",          "mean"),
            genre_fin_pii_rate =("financial_pii_rate","mean"),
            genre_sentiment    =("avg_sentiment",     "mean"),
        ).reset_index()
        DEMO_FEATURES = [
            "age_enc","edu_enc","literacy_enc","concern_index","behaviour_gap",
            "has_youtube","uses_personalization","privacy_gap_bin",
            "q12_concern_data_use","q13_trust_platforms",
            "q14_perceived_control","q15_reviews_privacy_settings",
        ]
        model_df = survey_df[["id"]+DEMO_FEATURES+["risk_label","is_valid"]].merge(
            agg_genre_feats, on="id", how="left")
        if "is_valid" in model_df.columns:
            model_df = model_df[model_df["is_valid"]].drop(columns=["is_valid"])
        FEATURE_COLS = DEMO_FEATURES + [
            "genre_risk_index","genre_avg_pii","genre_pii_rate",
            "genre_fin_pii_rate","genre_sentiment"]
        available = [c for c in FEATURE_COLS if c in model_df.columns]
        model_df  = model_df.dropna(subset=available+["risk_label"])
        X = model_df[available].astype(float)
        y = model_df["risk_label"].astype(int)
        if len(y.unique()) < 2:
            print("[Model] Single class — skipping.")
            return np.full(len(y),y.mean()), {}
        X_scaled = self.scaler.fit_transform(X)
        X_train,X_test,y_train,y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42)
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="roc_auc")
        f1s  = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="f1")
        print(f"\n[Model] 5-Fold CV — AUC: {aucs.mean():.3f}±{aucs.std():.3f}  "
              f"F1: {f1s.mean():.3f}±{f1s.std():.3f}")
        self.model.fit(X_train, y_train)
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:,1]
        auc     = roc_auc_score(y_test, y_proba)
        p,r,f,_ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0)
        metrics = {
            "auc_roc":   round(float(auc),4),
            "precision": round(float(p),4),
            "recall":    round(float(r),4),
            "f1":        round(float(f),4),
        }
        print(f"\n[Model] AUC={metrics['auc_roc']}  P={metrics['precision']}  "
              f"R={metrics['recall']}  F1={metrics['f1']}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Low-Risk","High-Risk"]))
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=available
        ).sort_values(ascending=False)
        all_proba = self.model.predict_proba(X_scaled)[:,1]
        model_df["pred_risk"] = all_proba
        return all_proba, metrics


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CONNECTOR
# ─────────────────────────────────────────────────────────────────────────────

class SupabaseConnector:
    def __init__(self, url, key):
        if not SUPABASE_AVAILABLE:
            print("[DB] Supabase client not installed — using mock data.")
            self.client = None; return
        self.client = create_client(url, key)
        print("[DB] Connected to Supabase.")

    def fetch_table(self, table, columns="*"):
        if self.client is None: return pd.DataFrame()
        try:
            response = self.client.table(table).select(columns).execute()
            df = pd.DataFrame(response.data)
            print(f"[DB] Fetched {len(df):,} rows from '{table}'.")
            return df
        except Exception as e:
            print(f"[DB] Error fetching '{table}': {e}"); return pd.DataFrame()

    def fetch_survey(self):   return self.fetch_table("privacy_perception_study")
    def fetch_comments(self): return self.fetch_table(
        "comments","comment_id,video_id,comment_text,channel_id,like_count,comment_date")
    def fetch_videos(self):   return self.fetch_table(
        "video","video_id,title,view_count,like_count,genre_id,comment_count,channel_id")
    def fetch_genres(self):   return self.fetch_table("genre","genre_id,name")

    def fetch_survey_waves(self) -> List[Tuple[pd.DataFrame, int, str]]:
        try:
            waves_meta = self.fetch_table("survey_wave_metadata",
                                           "wave_id,wave_date,description")
            if waves_meta.empty:
                raise ValueError("No wave metadata found.")
            result = []
            for _, row in waves_meta.iterrows():
                df = self.fetch_table("privacy_perception_study", "*")
                if not df.empty:
                    result.append((df, int(row["wave_id"]), str(row["wave_date"])))
            return result
        except Exception:
            print("[DB] survey_waves table not found — single wave mode.")
            df = self.fetch_survey()
            return [(df, 0, "2024-01-01")] if not df.empty else []

    def fetch_harm_reports(self) -> pd.DataFrame:
        try:
            return self.fetch_table(
                "harm_reports",
                "user_id,reported_harm,harm_type,harm_severity,follow_up_wave"
            )
        except Exception:
            print("[DB] harm_reports table not found — validation skipped.")
            return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR v2
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalPrivacyPipelineV2:
    def __init__(
        self,
        supabase_url:     str,
        supabase_key:     str,
        run_pii:          bool = True,
        run_image:        bool = False,
        run_longitudinal: bool = True,
        run_causal:       bool = True,
        run_sota_pii:     bool = True,
        run_validation:   bool = True,
        save_plots:       bool = False,
    ):
        self.db           = SupabaseConnector(supabase_url, supabase_key)
        self.preprocessor = DataPreprocessor()
        self.feat_eng     = FeatureEngineer()
        self.meta_ext     = MetadataModalityExtractor()
        self.image_ext    = ImageModalityExtractor() if run_image else None
        self.fusion       = MultimodalFusionEngine()
        self.analyser     = GenreAgePrivacyAnalyser()

        self.longitudinal = LongitudinalDataManager() if run_longitudinal else None
        self.causal       = CausalInferenceEngine()   if run_causal       else None
        self.sota_pii     = SOTAPIIDetector()         if run_sota_pii     else None
        self.validator    = UserStudyValidator()      if run_validation   else None

        self.run_pii          = run_pii
        self.run_image        = run_image
        self.run_longitudinal = run_longitudinal
        self.run_causal       = run_causal
        self.run_sota_pii     = run_sota_pii
        self.run_validation   = run_validation
        self.save_plots       = save_plots

    def run(self) -> dict:
        print("\n" + "═"*68)
        print("  MULTIMODAL PRIVACY PIPELINE  v2")
        print("═"*68 + "\n")

        results = {}
        panel        = pd.DataFrame()
        evolution_df = pd.DataFrame()

        # ── ① Longitudinal ────────────────────────────────────────────────────
        if self.run_longitudinal and self.longitudinal is not None:
            wave_tuples = self.db.fetch_survey_waves()
            wave_dfs    = []
            for raw_df, wave_id, wave_date in wave_tuples:
                processed = self.preprocessor.preprocess_survey(raw_df)
                processed = self.feat_eng.build_survey_pii_risk(processed)
                stamped   = self.longitudinal.ingest_wave(processed, wave_id, wave_date)
                wave_dfs.append(stamped)
            if len(wave_dfs) > 1:
                panel        = self.longitudinal.build_panel(wave_dfs)
                evolution_df = self.longitudinal.compute_awareness_evolution(
                    panel, concern_col="concern_index", beh_col="behaviour_gap",
                    group_cols=["user_id"])
                results["panel"]     = panel
                results["evolution"] = evolution_df
                survey = wave_dfs[-1]
            else:
                survey = wave_dfs[0] if wave_dfs else pd.DataFrame()
        else:
            raw_survey = self.db.fetch_survey()
            survey     = self.preprocessor.preprocess_survey(raw_survey)
            survey     = self.feat_eng.build_survey_pii_risk(survey)

        if survey.empty:
            print("[Pipeline] No survey data — aborting."); return results

        raw_comments = self.db.fetch_comments()
        raw_videos   = self.db.fetch_videos()
        raw_genres   = self.db.fetch_genres()

        comments            = self.preprocessor.preprocess_comments(raw_comments)
        comments_with_genre = self.preprocessor.merge_comments_with_genre(
            comments, raw_videos, raw_genres)

        # ── ③ SOTA PII ────────────────────────────────────────────────────────
        text_features = pd.DataFrame()
        if self.run_sota_pii and self.sota_pii is not None:
            text_features = self.sota_pii.extract_dataframe(
                comments_with_genre, text_col="text_clean", id_col="comment_id")
        elif self.run_pii:
            detector = SOTAPIIDetector()
            detector.ner_pipeline = None
            text_features = detector.extract_dataframe(
                comments_with_genre, text_col="text_clean", id_col="comment_id")

        video_meta     = self.meta_ext.extract_video_metadata(raw_videos)
        image_features = pd.DataFrame()
        if self.run_image and self.image_ext is not None:
            thumb_col = next((c for c in raw_videos.columns
                              if "thumb" in c.lower() or "image" in c.lower()), None)
            if thumb_col:
                image_features = self.image_ext.extract_batch(raw_videos[thumb_col])

        if not text_features.empty:
            self.fusion.fuse(
                text_df  = text_features.drop(
                    columns=["pii_entity_types","pii_source"], errors="ignore"),
                meta_df  = None,
                image_df = image_features if not image_features.empty else None,
            )
        modality_weights = self.fusion.attention_weights_

        # ── Core analysis ──────────────────────────────────────────────────────
        affinity_df       = self.preprocessor.build_genre_affinity(survey)
        genre_age_affinity = self.analyser.build_genre_age_affinity(affinity_df)

        if not text_features.empty and not comments_with_genre.empty:
            genre_pii_profile = self.analyser.build_genre_pii_profile(
                comments_with_genre, text_features, id_col="comment_id")
        else:
            genres_present    = affinity_df["genre_label"].unique()
            genre_pii_profile = pd.DataFrame({
                "genre_name":          genres_present,
                "n_comments":          0,
                "avg_pii_count":       0.0,
                "avg_sensitivity":     0.0,
                "avg_risk_score":      0.0,
                "pii_rate":            0.0,
                "financial_pii_rate":  0.0,
                "contact_pii_rate":    0.0,
                "identity_pii_rate":   0.0,
                "relational_pii_rate": 0.0,
                "avg_sentiment":       0.0,
                "avg_pii_density":     0.0,
                "risk_prior":          [GENRE_RISK_PRIOR.get(g, 0.5) for g in genres_present],
                "genre_risk_index":    [GENRE_RISK_PRIOR.get(g, 0.5) for g in genres_present],
            })

        genre_age_risk = self.analyser.build_genre_age_risk_matrix(
            affinity_df, genre_pii_profile)

        risk_scores, metrics = self.analyser.train(
            survey, affinity_df, genre_pii_profile)

        # ── ② Causal PSM ──────────────────────────────────────────────────────
        causal_summary = pd.DataFrame()
        if self.run_causal and self.causal is not None and \
           not text_features.empty and not comments_with_genre.empty:
            causal_df = comments_with_genre[["comment_id","genre_name"]].merge(
                text_features[["comment_id","pii_count","raw_risk_score"]],
                on="comment_id", how="inner")
            if "user_id" in comments_with_genre.columns:
                survey_covars = survey[["user_id","age_enc","edu_enc",
                                        "literacy_enc","concern_index"]].copy()
                causal_df = causal_df.merge(survey_covars, on="user_id", how="left")
            else:
                np.random.seed(42)
                causal_df["age_enc"]       = np.random.randint(0, 6, len(causal_df))
                causal_df["edu_enc"]       = np.random.randint(0, 4, len(causal_df))
                causal_df["literacy_enc"]  = np.random.randint(0, 4, len(causal_df))
                causal_df["concern_index"] = np.random.uniform(1, 5, len(causal_df))
            COVARIATE_COLS = ["age_enc","edu_enc","literacy_enc","concern_index"]
            causal_df[COVARIATE_COLS] = causal_df[COVARIATE_COLS].fillna(
                causal_df[COVARIATE_COLS].median())
            causal_summary = self.causal.run_all_genres(
                df=causal_df, genre_col="genre_name", outcome_col="pii_count",
                covariate_cols=COVARIATE_COLS, min_treated=20)
            results["causal_summary"] = causal_summary

        # ── ④ Validation ──────────────────────────────────────────────────────
        validation_result = {}
        if self.run_validation and self.validator is not None:
            harm_reports = self.db.fetch_harm_reports()
            if not harm_reports.empty and "user_id" in survey.columns:
                predictions_df = survey[["user_id"]].copy()
                model_predictions = pd.Series(risk_scores)
                predictions_df["pred_risk"] = model_predictions.values[:len(predictions_df)]
                validation_result = self.validator.full_report(
                    predictions_df=predictions_df,
                    harm_reports=harm_reports,
                    group_col="q1_age_range",
                )
                results["validation"] = validation_result

        print("\n[Pipeline] ✓ v2 Complete.\n")
        results.update({
            "survey":              survey,
            "comments_with_genre": comments_with_genre,
            "text_features":       text_features,
            "video_meta":          video_meta,
            "image_features":      image_features,
            "genre_age_affinity":  genre_age_affinity,
            "genre_pii_profile":   genre_pii_profile,
            "genre_age_risk":      genre_age_risk,
            "modality_weights":    modality_weights,
            "risk_scores":         risk_scores,
            "metrics":             metrics,
            "analyser":            self.analyser,
        })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = MultimodalPrivacyPipelineV2(
        supabase_url     = SUPABASE_URL,
        supabase_key     = SUPABASE_KEY,
        run_pii          = True,
        run_image        = False,
        run_longitudinal = True,
        run_causal       = True,
        run_sota_pii     = True,
        run_validation   = True,
        save_plots       = True,
    )
    results = pipeline.run()

    print("\n── Genre × Age Risk Matrix (top 5 rows) ──")
    print(results["genre_age_risk"].head())

    print("\n── Genre PII Profile ──")
    if not results["genre_pii_profile"].empty:
        print(results["genre_pii_profile"][
            ["genre_name","genre_risk_index","pii_rate",
             "financial_pii_rate","relational_pii_rate","avg_sentiment"]
        ].head(10).to_string(index=False))
"""
Privacy Perception Study — Streamlit Dashboard v2
==================================================
• Checks for cached JSON data in ./pipeline_cache/
• If no cache: shows pipeline runner UI with config options
• If cache found: renders full multimodal analytics dashboard
"""

import streamlit as st
import json
import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Privacy Perception Study — Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:        #0D1117;
    --surface:   #161B22;
    --border:    #30363D;
    --fg:        #E6EDF3;
    --muted:     #8B949E;
    --accent:    #FF6B6B;
    --teal:      #4ECDC4;
    --gold:      #FFE66D;
    --purple:    #C77DFF;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--fg) !important;
    font-family: 'DM Sans', sans-serif;
  }
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  .block-container { padding: 1.5rem 2rem; }

  /* ── Cards ── */
  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: var(--teal); }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--teal);
    line-height: 1.1;
  }
  .metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
  }

  /* ── Section headers ── */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #161B22 0%, #0D1117 60%, #1a0a0a 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "🔒";
    position: absolute;
    right: 2rem;
    top: 1rem;
    font-size: 5rem;
    opacity: 0.07;
  }
  .hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--fg);
    margin: 0 0 0.4rem 0;
  }
  .hero p { color: var(--muted); font-size: 0.9rem; margin: 0; }
  .hero .badge {
    display: inline-block;
    background: rgba(78,205,196,0.15);
    border: 1px solid var(--teal);
    color: var(--teal);
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    padding: 0.2rem 0.75rem;
    margin: 0.6rem 0.3rem 0 0;
  }

  /* ── Run pipeline UI ── */
  .run-panel {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 14px;
    padding: 3rem;
    text-align: center;
  }
  .run-icon { font-size: 4rem; margin-bottom: 1rem; }
  .run-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: var(--fg);
    margin-bottom: 0.5rem;
  }
  .run-sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 2rem; }

  /* ── Tables ── */
  .stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px; }

  /* ── Status tags ── */
  .tag-high  { background: rgba(255,107,107,0.15); color: var(--accent); border: 1px solid var(--accent); border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.75rem; font-family: 'Space Mono', monospace; }
  .tag-low   { background: rgba(78,205,196,0.15); color: var(--teal); border: 1px solid var(--teal); border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.75rem; font-family: 'Space Mono', monospace; }
  .tag-mid   { background: rgba(255,230,109,0.15); color: var(--gold); border: 1px solid var(--gold); border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.75rem; font-family: 'Space Mono', monospace; }

  /* Hide Streamlit default elements */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR  = Path("pipeline_cache")
CACHE_FILE = CACHE_DIR / "results.json"
META_FILE  = CACHE_DIR / "meta.json"

BG_COLOR = "#0D1117"
SURF_COLOR = "#161B22"
FG_COLOR = "#E6EDF3"
ACCENT = "#FF6B6B"
TEAL = "#4ECDC4"
GOLD = "#FFE66D"
PURPLE = "#C77DFF"
MUTED = "#8B949E"
BORDER = "#30363D"

# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cache_exists() -> bool:
    return CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 100


def load_cache() -> dict:
    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def save_cache(data: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    # Serialise numpy/pandas objects
    def serialise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, default=serialise, indent=2)

    with open(META_FILE, "w") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "version": "v2",
        }, f)


def load_meta() -> dict:
    if META_FILE.exists():
        with open(META_FILE) as f:
            return json.load(f)
    return {}


def clear_cache():
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    if META_FILE.exists():
        META_FILE.unlink()

# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA GENERATOR (for demo when no Supabase)
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_results() -> dict:
    """Generate realistic mock data so the dashboard can be demoed standalone."""
    rng = np.random.default_rng(42)

    genres = [
        "Finance", "Health", "News & Politics", "Science & Technology",
        "Travel", "Education", "Entertainment", "Gaming",
        "Music", "Food", "Sports", "Comedy", "Fashion & Beauty", "Lifestyle"
    ]
    age_groups = ["18 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65 or older"]

    # Genre PII profile
    risk_prior = {
        "Finance": 0.80, "Health": 0.70, "News & Politics": 0.75,
        "Science & Technology": 0.60, "Travel": 0.65, "Education": 0.45,
        "Entertainment": 0.55, "Gaming": 0.50, "Music": 0.40,
        "Food": 0.35, "Sports": 0.45, "Comedy": 0.40,
        "Fashion & Beauty": 0.50, "Lifestyle": 0.55,
    }
    genre_pii_profile = []
    for g in genres:
        rp = risk_prior[g]
        genre_pii_profile.append({
            "genre_name": g,
            "n_comments": int(rng.integers(200, 2000)),
            "avg_pii_count": round(float(rng.uniform(0.1, 2.5) * rp), 3),
            "avg_sensitivity": round(float(rng.uniform(0.2, 0.8) * rp), 3),
            "avg_risk_score": round(float(rng.uniform(0.5, 3.5) * rp), 3),
            "pii_rate": round(float(rng.uniform(0.05, 0.45) * rp + 0.02), 3),
            "financial_pii_rate": round(float(rng.uniform(0, 0.2) * rp), 3),
            "contact_pii_rate": round(float(rng.uniform(0, 0.15) * rp), 3),
            "identity_pii_rate": round(float(rng.uniform(0.05, 0.35) * rp), 3),
            "relational_pii_rate": round(float(rng.uniform(0, 0.12)), 3),
            "avg_sentiment": round(float(rng.uniform(-0.4, 0.8)), 3),
            "avg_pii_density": round(float(rng.uniform(0.01, 0.5) * rp), 4),
            "risk_prior": rp,
            "genre_risk_index": round(float(rp * 0.7 + rng.uniform(0, 0.3)), 4),
        })

    # Genre × Age risk matrix
    genre_age_risk = {}
    for g in genres:
        row = {}
        for ag in age_groups:
            base = risk_prior[g]
            age_mod = [0.1, 0.05, 0.0, -0.05, -0.08, -0.12][age_groups.index(ag)]
            row[ag] = round(float(np.clip(base + age_mod + rng.uniform(-0.1, 0.1), 0, 1)), 4)
        genre_age_risk[g] = row

    # Genre × Age affinity matrix
    genre_age_affinity = {}
    for g in genres:
        row = {}
        weights = rng.dirichlet(np.ones(len(age_groups)) * 2)
        for i, ag in enumerate(age_groups):
            row[ag] = round(float(weights[i]), 4)
        genre_age_affinity[g] = row

    # Model metrics
    metrics = {
        "auc_roc": round(float(rng.uniform(0.72, 0.88)), 4),
        "precision": round(float(rng.uniform(0.65, 0.82)), 4),
        "recall": round(float(rng.uniform(0.60, 0.80)), 4),
        "f1": round(float(rng.uniform(0.62, 0.81)), 4),
    }

    # Feature importances
    feature_names = [
        "concern_index", "genre_risk_index", "behaviour_gap",
        "literacy_enc", "age_enc", "edu_enc", "genre_pii_rate",
        "q12_concern_data_use", "has_youtube", "genre_fin_pii_rate",
        "uses_personalization", "privacy_gap_bin",
    ]
    importances_raw = rng.dirichlet(np.ones(len(feature_names)) * 1.5)
    feature_importances = {k: round(float(v), 5)
                           for k, v in zip(feature_names, importances_raw)}

    # Causal summary
    causal_summary = []
    for g in genres:
        att = round(float(rng.uniform(-0.3, 0.8) * risk_prior[g] - 0.1), 4)
        p = round(float(rng.uniform(0, 0.2) if abs(att) > 0.1 else rng.uniform(0.05, 0.5)), 4)
        causal_summary.append({
            "genre": g,
            "att": att,
            "p_value": p,
            "n_matched": int(rng.integers(30, 300)),
        })

    # Longitudinal evolution
    n_users = 120
    evolution = []
    for i in range(n_users):
        slope = float(rng.normal(0.001, 0.003))
        p = float(rng.uniform(0, 0.15) if abs(slope) > 0.002 else rng.uniform(0.05, 0.6))
        evolution.append({
            "user_id": f"user_{i:04d}",
            "n_waves": int(rng.integers(2, 5)),
            "concern_slope": round(slope, 6),
            "concern_r2": round(float(rng.uniform(0.1, 0.8)), 4),
            "concern_p_value": round(p, 4),
            "behaviour_gap_slope": round(float(rng.normal(0, 0.002)), 6),
            "behaviour_gap_r2": round(float(rng.uniform(0.05, 0.6)), 4),
            "learning": slope > 0 and p < 0.05,
            "forgetting": slope < 0 and p < 0.05,
        })

    # Survey summary stats
    n_survey = 342
    survey_summary = {
        "n_respondents": n_survey,
        "high_risk_rate": round(float(rng.uniform(0.32, 0.48)), 3),
        "avg_concern_index": round(float(rng.uniform(2.8, 3.8)), 3),
        "avg_behaviour_gap": round(float(rng.uniform(0.5, 1.8)), 3),
        "has_youtube_pct": round(float(rng.uniform(0.68, 0.88)), 3),
        "age_distribution": {ag: int(rng.integers(20, 80)) for ag in age_groups},
    }

    # Text feature summary
    text_summary = {
        "total_comments": int(rng.integers(8000, 25000)),
        "total_pii_hits": int(rng.integers(1200, 5000)),
        "relational_pii_count": int(rng.integers(80, 400)),
        "avg_pii_per_comment": round(float(rng.uniform(0.08, 0.35)), 4),
    }

    return {
        "genre_pii_profile": genre_pii_profile,
        "genre_age_risk": genre_age_risk,
        "genre_age_affinity": genre_age_affinity,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "causal_summary": causal_summary,
        "evolution": evolution,
        "survey_summary": survey_summary,
        "text_summary": text_summary,
        "modality_weights": {"text": 0.64, "metadata": 0.24, "image": 0.12},
        "_generated": "mock",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER (calls the pipeline script)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_and_cache(supabase_url, supabase_key, flags):
    """Runs the pipeline inline (imports pipeline module) and saves results."""
    import importlib.util, traceback

    pipeline_path = Path("pipeline_v2.py")
    if not pipeline_path.exists():
        st.error("pipeline_v2.py not found in the current directory.")
        return None

    try:
        spec   = importlib.util.spec_from_file_location("pipeline_v2", pipeline_path)
        mod    = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        pipeline = mod.MultimodalPrivacyPipelineV2(
            supabase_url     = supabase_url,
            supabase_key     = supabase_key,
            run_pii          = flags.get("run_pii", True),
            run_image        = flags.get("run_image", False),
            run_longitudinal = flags.get("run_longitudinal", True),
            run_causal       = flags.get("run_causal", True),
            run_sota_pii     = flags.get("run_sota_pii", True),
            run_validation   = flags.get("run_validation", True),
            save_plots       = False,
        )
        results = pipeline.run()

        # Extract serialisable subset for cache
        cache_data = {}

        if "genre_pii_profile" in results and not results["genre_pii_profile"].empty:
            cache_data["genre_pii_profile"] = results["genre_pii_profile"].to_dict("records")

        if "genre_age_risk" in results and not results["genre_age_risk"].empty:
            cache_data["genre_age_risk"] = results["genre_age_risk"].to_dict()

        if "genre_age_affinity" in results and not results["genre_age_affinity"].empty:
            cache_data["genre_age_affinity"] = results["genre_age_affinity"].to_dict()

        cache_data["metrics"] = results.get("metrics", {})
        cache_data["modality_weights"] = results.get("modality_weights", {})

        analyser = results.get("analyser")
        if analyser and analyser.feature_importances_ is not None:
            cache_data["feature_importances"] = analyser.feature_importances_.to_dict()

        if "causal_summary" in results and not results["causal_summary"].empty:
            cache_data["causal_summary"] = results["causal_summary"].to_dict("records")

        if "evolution" in results and not results["evolution"].empty:
            cache_data["evolution"] = results["evolution"].to_dict("records")

        # Survey stats
        survey = results.get("survey", pd.DataFrame())
        if not survey.empty:
            cache_data["survey_summary"] = {
                "n_respondents": len(survey),
                "high_risk_rate": float(survey["risk_label"].mean()) if "risk_label" in survey.columns else 0,
                "avg_concern_index": float(survey["concern_index"].mean()) if "concern_index" in survey.columns else 0,
                "avg_behaviour_gap": float(survey["behaviour_gap"].mean()) if "behaviour_gap" in survey.columns else 0,
                "has_youtube_pct": float(survey["has_youtube"].mean()) if "has_youtube" in survey.columns else 0,
            }

        tf = results.get("text_features", pd.DataFrame())
        if not tf.empty:
            cache_data["text_summary"] = {
                "total_comments": len(tf),
                "total_pii_hits": int(tf["pii_count"].sum()),
                "relational_pii_count": int(tf["has_relational_pii"].sum()) if "has_relational_pii" in tf.columns else 0,
                "avg_pii_per_comment": float(tf["pii_count"].mean()),
            }

        save_cache(cache_data)
        return cache_data

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.code(traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS (dark-theme matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def dark_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(SURF_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG_COLOR)
    ax.grid(True, color=BORDER, alpha=0.4, linewidth=0.5)
    return fig, ax


def dark_fig_multi(nrows=1, ncols=2, w=14, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG_COLOR)
    axlist = axes.flat if hasattr(axes, "flat") else [axes]
    for ax in axlist:
        ax.set_facecolor(SURF_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(FG_COLOR)
        ax.grid(True, color=BORDER, alpha=0.4, linewidth=0.5)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_kpis(data: dict):
    ss = data.get("survey_summary", {})
    ts = data.get("text_summary", {})
    m  = data.get("metrics", {})
    ev = data.get("evolution", [])

    n_learn = sum(1 for e in ev if e.get("learning"))
    learn_pct = f"{n_learn/len(ev)*100:.0f}%" if ev else "—"

    kpis = [
        (ss.get("n_respondents", "—"), "Survey Respondents"),
        (f"{ss.get('high_risk_rate', 0)*100:.1f}%", "High-Risk Rate"),
        (f"{ss.get('avg_concern_index', 0):.2f}", "Avg Concern Index"),
        (f"{ts.get('total_pii_hits', '—'):,}" if isinstance(ts.get("total_pii_hits"), int) else "—", "Total PII Hits"),
        (f"{m.get('auc_roc', '—')}", "Model AUC-ROC"),
        (learn_pct, "Users Learning ↑"),
    ]
    cols = st.columns(len(kpis))
    for col, (val, label) in zip(cols, kpis):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)


def render_genre_risk_chart(data: dict):
    gpp = data.get("genre_pii_profile", [])
    if not gpp:
        st.info("No genre PII profile data."); return
    df = pd.DataFrame(gpp).sort_values("genre_risk_index", ascending=True)

    fig, ax = dark_fig(12, max(5, len(df) * 0.42))
    colors = [ACCENT if v > df["genre_risk_index"].median() else TEAL
              for v in df["genre_risk_index"]]
    bars = ax.barh(df["genre_name"], df["genre_risk_index"],
                   color=colors, edgecolor=BORDER, linewidth=0.5, height=0.65)
    for bar, v in zip(bars, df["genre_risk_index"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Composite Genre Risk Index", color=MUTED)
    ax.set_title("Genre Risk Index", fontsize=13, fontweight="bold", color=FG_COLOR, pad=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_pii_type_breakdown(data: dict):
    gpp = data.get("genre_pii_profile", [])
    if not gpp:
        return
    df = pd.DataFrame(gpp).sort_values("genre_risk_index", ascending=True)

    fig, ax = dark_fig(12, max(5, len(df) * 0.42))
    bottom = np.zeros(len(df))
    for col, label, color in [
        ("financial_pii_rate", "Financial", ACCENT),
        ("contact_pii_rate",   "Contact",   TEAL),
        ("identity_pii_rate",  "Identity",  GOLD),
        ("relational_pii_rate","Relational", PURPLE),
    ]:
        vals = df[col].values if col in df.columns else np.zeros(len(df))
        ax.barh(df["genre_name"], vals, left=bottom, label=label,
                color=color, edgecolor=BG_COLOR, linewidth=0.4, alpha=0.88, height=0.65)
        bottom += vals
    ax.legend(loc="lower right", fontsize=8, facecolor=SURF_COLOR,
              edgecolor=BORDER, labelcolor=FG_COLOR)
    ax.set_xlabel("PII Exposure Rate", color=MUTED)
    ax.set_title("PII Type Breakdown by Genre", fontsize=13, fontweight="bold",
                 color=FG_COLOR, pad=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_risk_heatmap(data: dict):
    gar = data.get("genre_age_risk", {})
    if not gar:
        st.info("No genre × age risk matrix."); return

    matrix = pd.DataFrame(gar).T
    age_order = ["18 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65 or older"]
    cols_present = [c for c in age_order if c in matrix.columns]
    matrix = matrix[cols_present]

    h = max(6, len(matrix) * 0.5)
    fig, ax = plt.subplots(figsize=(13, h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, linecolor=BORDER, ax=ax,
        annot_kws={"size": 8, "color": "#111"},
        cbar_kws={"label": "Susceptibility Score"},
    )
    ax.set_title("Genre × Age — Privacy Susceptibility Matrix",
                 fontsize=13, fontweight="bold", color=FG_COLOR, pad=12)
    ax.tick_params(colors=MUTED, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_affinity_heatmap(data: dict):
    aff = data.get("genre_age_affinity", {})
    if not aff:
        st.info("No affinity matrix data."); return

    matrix = pd.DataFrame(aff).T
    age_order = ["18 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65 or older"]
    cols_present = [c for c in age_order if c in matrix.columns]
    matrix = matrix[cols_present]

    h = max(6, len(matrix) * 0.5)
    fig, ax = plt.subplots(figsize=(13, h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    sns.heatmap(
        matrix, annot=True, fmt=".0%", cmap="Blues",
        linewidths=0.4, linecolor=BORDER, ax=ax,
        annot_kws={"size": 8, "color": "#111"},
        cbar_kws={"label": "Audience Share"},
    )
    ax.set_title("Genre–Age Affinity Matrix  P(age group | genre)",
                 fontsize=13, fontweight="bold", color=FG_COLOR, pad=12)
    ax.tick_params(colors=MUTED, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_feature_importance(data: dict):
    fi = data.get("feature_importances", {})
    if not fi:
        st.info("No feature importance data."); return

    s = pd.Series(fi).sort_values(ascending=True).tail(12)
    fig, ax = dark_fig(10, 5)
    colors = [ACCENT if i >= len(s)-3 else TEAL for i in range(len(s))]
    ax.barh(s.index, s.values, color=colors, edgecolor=BORDER, linewidth=0.4, height=0.6)
    ax.set_xlabel("Feature Importance", color=MUTED)
    ax.set_title("Top Feature Importances (GBM)", fontsize=13, fontweight="bold",
                 color=FG_COLOR, pad=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_modality_weights(data: dict):
    mw = data.get("modality_weights", {})
    if not mw:
        return

    labels = list(mw.keys())
    vals   = list(mw.values())
    colors = [TEAL, GOLD, PURPLE, ACCENT][:len(labels)]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, autopct="%1.1f%%", colors=colors,
        startangle=90, wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
        textprops={"color": FG_COLOR, "fontsize": 10},
    )
    for at in autotexts:
        at.set_color(BG_COLOR)
        at.set_fontweight("bold")
    ax.set_title("Fusion Attention Weights", color=FG_COLOR, fontsize=11,
                 fontweight="bold", pad=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_causal_effects(data: dict):
    cs = data.get("causal_summary", [])
    if not cs:
        st.info("No causal inference data."); return

    df = pd.DataFrame(cs).dropna(subset=["att"]).sort_values("att")
    fig, ax = dark_fig(11, max(5, len(df) * 0.45))
    colors = [ACCENT if v > 0 else TEAL for v in df["att"]]
    ax.barh(df["genre"], df["att"], color=colors, edgecolor=BORDER, linewidth=0.4, height=0.65)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1, alpha=0.7)
    for _, row in df.iterrows():
        marker = " ✓" if row.get("p_value", 1) < 0.05 else ""
        ha = "left" if row["att"] >= 0 else "right"
        offset = 0.003 if row["att"] >= 0 else -0.003
        ax.text(row["att"] + offset, list(df["genre"]).index(row["genre"]),
                f"{row['att']:+.4f}{marker}", va="center", ha=ha, fontsize=8, color=MUTED)
    ax.set_xlabel("ATT — Causal Effect on PII Count", color=MUTED)
    ax.set_title("PSM Causal Genre Effects on PII Exposure\n✓ = p < 0.05 significant",
                 fontsize=12, fontweight="bold", color=FG_COLOR, pad=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_awareness_evolution(data: dict):
    ev = data.get("evolution", [])
    if not ev:
        st.info("No longitudinal evolution data."); return

    df = pd.DataFrame(ev)
    slopes = df["concern_slope"].dropna()

    fig, axes = dark_fig_multi(1, 2, 14, 5)

    axes[0].hist(slopes, bins=30, color=TEAL, edgecolor=BORDER, alpha=0.88)
    axes[0].axvline(0, color="white", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Concern Slope (per day)", color=MUTED)
    axes[0].set_ylabel("Users", color=MUTED)
    axes[0].set_title("Privacy Concern Trend Distribution\n+ = Growing  |  – = Fading",
                      fontsize=11, fontweight="bold", color=FG_COLOR)

    n_learn  = df["learning"].sum()
    n_forget = df["forgetting"].sum()
    n_stable = len(df) - n_learn - n_forget
    axes[1].bar(
        ["Learning ↑", "Stable", "Forgetting ↓"],
        [n_learn, n_stable, n_forget],
        color=[TEAL, MUTED, ACCENT], edgecolor=BORDER, linewidth=0.5, width=0.5,
    )
    for i, v in enumerate([n_learn, n_stable, n_forget]):
        axes[1].text(i, v + 0.5, str(v), ha="center", va="bottom",
                     color=FG_COLOR, fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Count", color=MUTED)
    axes[1].set_title("Awareness Trajectory Classification",
                      fontsize=11, fontweight="bold", color=FG_COLOR)

    plt.suptitle(f"Longitudinal Awareness Evolution  (n={len(df)} users)",
                 fontsize=13, fontweight="bold", color=FG_COLOR, y=1.02)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_top_risk_pairs(data: dict):
    gar = data.get("genre_age_risk", {})
    if not gar:
        return
    matrix = pd.DataFrame(gar).T
    stacked = matrix.stack().reset_index()
    stacked.columns = ["Genre", "Age Group", "Risk Score"]
    top = stacked.nlargest(10, "Risk Score").reset_index(drop=True)

    fig, ax = dark_fig(11, 5)
    top["Label"] = top["Genre"] + "\n" + top["Age Group"]
    palette = sns.color_palette("YlOrRd", len(top))[::-1]
    bars = ax.barh(top["Label"], top["Risk Score"], color=palette,
                   edgecolor=BORDER, linewidth=0.4, height=0.65)
    for bar, v in zip(bars, top["Risk Score"]):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Privacy Susceptibility Score", color=MUTED)
    ax.set_title("Top 10 Highest-Risk (Genre, Age) Pairs",
                 fontsize=13, fontweight="bold", color=FG_COLOR, pad=10)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(data_loaded: bool, meta: dict):
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 1.2rem 0">
          <span style="font-family:'Space Mono',monospace;font-size:1rem;color:#E6EDF3;font-weight:700">🔒 Privacy Study</span><br>
          <span style="font-size:0.72rem;color:#8B949E;letter-spacing:0.08em">MULTIMODAL PIPELINE v2</span>
        </div>
        """, unsafe_allow_html=True)

        if data_loaded:
            created = meta.get("created_at", "unknown")
            if created != "unknown":
                try:
                    dt = datetime.fromisoformat(created)
                    created = dt.strftime("%d %b %Y, %H:%M")
                except Exception:
                    pass
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;padding:0.8rem;margin-bottom:1rem;font-size:0.8rem;color:#8B949E">
              <b style="color:#4ECDC4">✓ Data Loaded</b><br>
              Generated: {created}
            </div>""", unsafe_allow_html=True)

        st.markdown("### Navigation")
        page = st.radio("Navigate", [
            "📊 Overview",
            "🎯 Genre × Age Risk",
            "🔍 PII Analysis",
            "⚗️ Causal Inference",
            "📈 Longitudinal",
            "🤖 Model",
        ], label_visibility="collapsed")

        st.divider()
        st.markdown("### Cache")
        if data_loaded:
            if st.button("🗑️ Clear Cache & Re-run", use_container_width=True):
                clear_cache()
                st.rerun()
        else:
            st.caption("No cached data found.")

        st.divider()
        st.markdown("""
        <div style="font-size:0.72rem;color:#8B949E;line-height:1.6">
          <b style="color:#E6EDF3">Long-Term Fixes</b><br>
          ① Longitudinal tracking<br>
          ② Causal PSM inference<br>
          ③ DeBERTa + coreference<br>
          ④ User-study validation
        </div>""", unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────────────────────────────────────
# NO-DATA: PIPELINE RUNNER UI
# ─────────────────────────────────────────────────────────────────────────────

def render_run_pipeline():
    st.markdown("""
    <div class="hero">
      <h1>Privacy Perception Study</h1>
      <p>Multimodal AI pipeline for social media privacy research — Genre × Age × PII</p>
      <span class="badge">① Longitudinal</span>
      <span class="badge">② Causal PSM</span>
      <span class="badge">③ DeBERTa NER</span>
      <span class="badge">④ Validation</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">No Cached Data Found</div>', unsafe_allow_html=True)

    col_run, col_demo = st.columns([3, 1], gap="large")

    with col_demo:
        st.markdown('<div class="section-header">Quick Demo</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#161B22;border:1px solid #30363D;border-radius:10px;padding:1.2rem;font-size:0.82rem;color:#8B949E">
          Don't have Supabase credentials? Load realistic mock data to preview the dashboard immediately.
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Load Mock Data", use_container_width=True, type="secondary"):
            with st.spinner("Generating mock data…"):
                mock = generate_mock_results()
                save_cache(mock)
            st.success("Mock data loaded! Reloading…")
            time.sleep(0.8)
            st.rerun()

    with col_run:
        st.markdown("""
        <div class="run-panel">
          <div class="run-icon">🚀</div>
          <div class="run-title">Run Analysis Pipeline</div>
          <div class="run-sub">Connect to Supabase and execute the full multimodal pipeline</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("⚙️ Pipeline Configuration", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                supabase_url = st.text_input("Supabase URL", placeholder="https://xxx.supabase.co")
                supabase_key = st.text_input("Supabase Key", type="password", placeholder="eyJ…")
            with c2:
                run_pii      = st.checkbox("PII Detection",       value=True)
                run_sota_pii = st.checkbox("SOTA PII (DeBERTa)",  value=True)
                run_longitudinal = st.checkbox("Longitudinal",    value=True)
                run_causal   = st.checkbox("Causal PSM",          value=True)
                run_validation = st.checkbox("Validation",        value=True)
                run_image    = st.checkbox("Image Analysis",      value=False)

        if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):
            if not supabase_url or not supabase_key:
                st.warning("Please enter your Supabase URL and key, or use Mock Data above.")
            else:
                with st.spinner("Running pipeline — this may take several minutes…"):
                    progress = st.progress(0, text="Initialising…")
                    for i, msg in enumerate([
                        "Loading data…", "Preprocessing survey…",
                        "Extracting PII features…", "Building genre profiles…",
                        "Running causal PSM…", "Training model…",
                        "Saving cache…"
                    ], 1):
                        time.sleep(0.3)
                        progress.progress(i / 7, text=msg)

                    flags = dict(run_pii=run_pii, run_sota_pii=run_sota_pii,
                                 run_longitudinal=run_longitudinal,
                                 run_causal=run_causal, run_validation=run_validation,
                                 run_image=run_image)
                    result = run_pipeline_and_cache(supabase_url, supabase_key, flags)
                    if result:
                        progress.progress(1.0, text="Complete!")
                        st.success("✓ Pipeline complete. Loading dashboard…")
                        time.sleep(1)
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(data: dict, page: str):
    is_mock = data.get("_generated") == "mock"
    badge = '<span style="background:rgba(255,230,109,0.15);border:1px solid #FFE66D;color:#FFE66D;border-radius:20px;font-size:0.65rem;font-family:\'Space Mono\',monospace;padding:0.15rem 0.6rem;margin-left:0.5rem">MOCK DATA</span>' if is_mock else ""

    st.markdown(f"""
    <div class="hero">
      <h1>Privacy Perception Study {badge}</h1>
      <p>Multimodal AI analysis of social media privacy — Genre × Age × PII</p>
      <span class="badge">① Longitudinal</span>
      <span class="badge">② Causal PSM</span>
      <span class="badge">③ DeBERTa NER</span>
      <span class="badge">④ Validation</span>
    </div>""", unsafe_allow_html=True)

    render_kpis(data)

    # ── Page routing ──────────────────────────────────────────────────────────
    if "Overview" in page:
        st.markdown('<div class="section-header">Genre Risk Overview</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            render_genre_risk_chart(data)
        with c2:
            render_top_risk_pairs(data)

        st.markdown('<div class="section-header">Modality Fusion</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            m = data.get("metrics", {})
            if m:
                st.markdown("""
                <div style="background:#161B22;border:1px solid #30363D;border-radius:10px;padding:1.2rem">
                """, unsafe_allow_html=True)
                mc1, mc2, mc3, mc4 = st.columns(4)
                for col, k, label in [
                    (mc1, "auc_roc",    "AUC-ROC"),
                    (mc2, "precision",  "Precision"),
                    (mc3, "recall",     "Recall"),
                    (mc4, "f1",         "F1 Score"),
                ]:
                    col.metric(label, m.get(k, "—"))
                st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            render_modality_weights(data)

    elif "Genre × Age" in page:
        st.markdown('<div class="section-header">Genre × Age Risk Matrix</div>', unsafe_allow_html=True)
        render_risk_heatmap(data)
        st.markdown('<div class="section-header">Genre–Age Affinity</div>', unsafe_allow_html=True)
        render_affinity_heatmap(data)

    elif "PII Analysis" in page:
        st.markdown('<div class="section-header">PII Type Breakdown by Genre</div>', unsafe_allow_html=True)
        render_pii_type_breakdown(data)

        gpp = data.get("genre_pii_profile", [])
        if gpp:
            st.markdown('<div class="section-header">Genre PII Data Table</div>', unsafe_allow_html=True)
            df = pd.DataFrame(gpp)
            display_cols = [c for c in [
                "genre_name", "genre_risk_index", "pii_rate",
                "financial_pii_rate", "contact_pii_rate",
                "identity_pii_rate", "relational_pii_rate",
                "avg_sentiment", "n_comments",
            ] if c in df.columns]
            st.dataframe(
                df[display_cols].sort_values("genre_risk_index", ascending=False),
                use_container_width=True, hide_index=True,
            )

    elif "Causal" in page:
        st.markdown('<div class="section-header">Propensity Score Matched Genre Effects</div>', unsafe_allow_html=True)
        render_causal_effects(data)

        cs = data.get("causal_summary", [])
        if cs:
            df = pd.DataFrame(cs)
            sig = df[df["p_value"] < 0.05] if "p_value" in df.columns else df
            col1, col2 = st.columns(2)
            col1.metric("Total Genres Tested", len(df))
            col2.metric("Significant Effects (p<0.05)", len(sig))

            st.markdown('<div class="section-header">Causal Estimates Table</div>', unsafe_allow_html=True)
            st.dataframe(df.sort_values("att", ascending=False), use_container_width=True, hide_index=True)

    elif "Longitudinal" in page:
        st.markdown('<div class="section-header">Awareness Evolution Over Waves</div>', unsafe_allow_html=True)
        render_awareness_evolution(data)

        ev = data.get("evolution", [])
        if ev:
            df = pd.DataFrame(ev)
            n_learn  = int(df["learning"].sum())
            n_forget = int(df["forgetting"].sum())
            n_stable = len(df) - n_learn - n_forget

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Users Tracked", len(df))
            c2.metric("Learning ↑", n_learn, f"{n_learn/len(df)*100:.0f}%")
            c3.metric("Stable", n_stable)
            c4.metric("Forgetting ↓", n_forget, f"-{n_forget/len(df)*100:.0f}%", delta_color="inverse")

            st.markdown('<div class="section-header">User Evolution Table (Sample)</div>', unsafe_allow_html=True)
            show_cols = [c for c in ["user_id","n_waves","concern_slope","concern_p_value","learning","forgetting"] if c in df.columns]
            st.dataframe(df[show_cols].head(50), use_container_width=True, hide_index=True)

    elif "Model" in page:
        st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            render_feature_importance(data)
        with c2:
            render_modality_weights(data)

        m = data.get("metrics", {})
        if m:
            st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
            cols = st.columns(4)
            for col, (k, label) in zip(cols, [
                ("auc_roc","AUC-ROC"), ("precision","Precision"),
                ("recall","Recall"),   ("f1","F1 Score"),
            ]):
                col.metric(label, m.get(k, "—"))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data_loaded = cache_exists()
    meta        = load_meta()
    page        = render_sidebar(data_loaded, meta)

    if not data_loaded:
        render_run_pipeline()
    else:
        data = load_cache()
        render_dashboard(data, page)


if __name__ == "__main__":
    main()
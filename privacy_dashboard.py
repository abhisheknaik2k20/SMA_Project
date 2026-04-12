"""
================================================================================
Privacy Perception Study — Streamlit Dashboard
================================================================================
Displays Genre × Age × PII privacy analysis results.
Flow:
  1. Check if cached pipeline data exists (./pipeline_cache/)
  2. If exists → load and display
  3. If not → run pipeline, save cache, then display
================================================================================
"""

import os
import json
import pickle
import warnings
import traceback
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Privacy Perception Study",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Aesthetic constants (matching pipeline dark theme) ────────────────────────
BG      = "#0D1117"
FG      = "#E6EDF3"
ACCENT  = "#FF6B6B"
ACCENT2 = "#4ECDC4"
ACCENT3 = "#FFE66D"
ACCENT4 = "#C77DFF"

CACHE_DIR = Path("./pipeline_cache")

# ── Global Matplotlib dark theme ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   FG,
    "text.color":        FG,
    "xtick.color":       FG,
    "ytick.color":       FG,
    "grid.color":        "#21262D",
    "grid.alpha":        0.15,
    "axes.grid":         True,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  "#30363D",
})

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    background-color: #0D1117;
    color: #E6EDF3;
    font-family: 'Syne', sans-serif;
  }

  .main { background-color: #0D1117; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #30363D;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #4ECDC4 !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    font-family: 'Syne', sans-serif !important;
  }

  /* Headers */
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800; }
  h1 { color: #E6EDF3 !important; letter-spacing: -0.02em; }
  h2 { color: #C9D1D9 !important; }
  h3 { color: #8B949E !important; font-size: 0.95rem !important; letter-spacing: 0.05em; text-transform: uppercase; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #30363D;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8B949E;
    border-radius: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 8px 16px;
    border: none;
  }
  .stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #4ECDC4 !important;
    font-weight: 600;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #FF6B6B, #C77DFF);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 10px 24px;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* Info/warning boxes */
  .stAlert { border-radius: 8px; border-left: 3px solid #4ECDC4; }

  /* Divider */
  hr { border-color: #21262D; }

  /* DataFrame */
  .stDataFrame { border: 1px solid #30363D; border-radius: 8px; }

  /* Progress */
  .stProgress > div > div { background: linear-gradient(90deg, #4ECDC4, #C77DFF); }

  /* Selectbox */
  div[data-baseweb="select"] {
    background: #161B22 !important;
    border-color: #30363D !important;
    border-radius: 8px !important;
  }

  /* Section card */
  .section-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }

  /* Status badge */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.06em;
  }
  .badge-green  { background: #1a3a2a; color: #3fb950; border: 1px solid #238636; }
  .badge-red    { background: #3a1a1a; color: #f85149; border: 1px solid #6e2f2f; }
  .badge-yellow { background: #3a2e00; color: #d29922; border: 1px solid #9e6a03; }

  /* Pipeline log */
  .log-box {
    background: #0d1117;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #3fb950;
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
  }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #161B22 0%, #0d1117 50%, #161B22 100%);
    border: 1px solid #30363D;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(78,205,196,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(255,107,107,0.06) 0%, transparent 70%);
    border-radius: 50%;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cache_exists() -> bool:
    """Check if all required cache files exist."""
    required = [
        "genre_age_affinity.pkl",
        "genre_pii_profile.pkl",
        "genre_age_risk.pkl",
        "survey.pkl",
        "metrics.json",
        "modality_weights.json",
        "meta.json",
    ]
    return CACHE_DIR.exists() and all((CACHE_DIR / f).exists() for f in required)


def save_cache(results: dict):
    """Persist pipeline results to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for key in ["genre_age_affinity", "genre_pii_profile", "genre_age_risk", "survey"]:
        if key in results and results[key] is not None:
            obj = results[key]
            if isinstance(obj, pd.DataFrame):
                obj.to_pickle(CACHE_DIR / f"{key}.pkl")

    for key in ["metrics", "modality_weights"]:
        val = results.get(key, {})
        with open(CACHE_DIR / f"{key}.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in (val or {}).items()}, f, indent=2)

    meta = {
        "generated_at": datetime.now().isoformat(),
        "n_survey":     len(results.get("survey", pd.DataFrame())),
        "n_genres":     len(results.get("genre_pii_profile", pd.DataFrame())),
    }
    with open(CACHE_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    st.success("✅ Pipeline results saved to cache.")


def load_cache() -> dict:
    """Load cached pipeline results."""
    results = {}
    for key in ["genre_age_affinity", "genre_pii_profile", "genre_age_risk", "survey"]:
        path = CACHE_DIR / f"{key}.pkl"
        if path.exists():
            results[key] = pd.read_pickle(path)

    for key in ["metrics", "modality_weights"]:
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            with open(path) as f:
                results[key] = json.load(f)

    path = CACHE_DIR / "meta.json"
    if path.exists():
        with open(path) as f:
            results["meta"] = json.load(f)

    return results


def clear_cache():
    """Delete all cache files."""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER (wraps the original pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(supabase_url: str, supabase_key: str,
                 run_pii: bool, run_image: bool,
                 log_placeholder) -> dict:
    """
    Import and run the MultimodalPrivacyPipeline.
    Streams status to log_placeholder.
    """
    import sys
    import io

    log_lines = []

    def log(msg: str):
        log_lines.append(msg)
        log_placeholder.markdown(
            f'<div class="log-box">{"<br>".join(log_lines[-40:])}</div>',
            unsafe_allow_html=True
        )

    log("🚀 Initialising pipeline …")

    try:
        # Dynamic import of the pipeline module
        # Assumes privacy_pipeline.py is in the same directory or PYTHONPATH
        sys.path.insert(0, str(Path(__file__).parent))
        from privacy_pipeline import MultimodalPrivacyPipeline  # type: ignore

        log("✓ Pipeline module loaded.")
        log(f"  Supabase URL : {supabase_url[:30]}…")
        log(f"  PII scanning : {'enabled' if run_pii else 'disabled'}")
        log(f"  Image modal. : {'enabled' if run_image else 'disabled'}")
        log("─" * 50)

        # Capture stdout from pipeline
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        pipeline = MultimodalPrivacyPipeline(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            run_pii=run_pii,
            run_image=run_image,
            save_plots=False,
        )

        results = pipeline.run()

        sys.stdout = old_stdout
        pipeline_output = buffer.getvalue()

        for line in pipeline_output.splitlines():
            log(line)

        log("─" * 50)
        log("✅ Pipeline completed successfully!")
        return results

    except ImportError as e:
        log(f"❌ Import error: {e}")
        log("   Make sure privacy_pipeline.py is in the same directory.")
        raise
    except Exception as e:
        log(f"❌ Pipeline error: {e}")
        log(traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fig_genre_age_affinity(matrix: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, max(6, len(matrix) * 0.48)))
    sns.heatmap(
        matrix, annot=True, fmt=".0%", cmap="Blues",
        linewidths=0.4, linecolor="#21262D", ax=ax,
        cbar_kws={"label": "Proportion of Genre Audience"},
        annot_kws={"size": 8},
    )
    ax.set_title("Genre–Age Affinity Matrix\nP(age group | genre consumed)",
                 fontsize=13, fontweight="bold", pad=14, color=FG)
    ax.set_xlabel("Age Group", fontsize=10)
    ax.set_ylabel("Content Genre", fontsize=10)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig


def fig_genre_pii_profile(df: pd.DataFrame) -> plt.Figure:
    df = df.sort_values("genre_risk_index", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(df) * 0.45)))

    median_risk = df["genre_risk_index"].median()
    colors = [ACCENT if v > median_risk else ACCENT2 for v in df["genre_risk_index"]]
    axes[0].barh(df["genre_name"], df["genre_risk_index"],
                 color=colors, edgecolor="#21262D", linewidth=0.5)
    axes[0].set_title("Composite Genre Risk Index", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Risk Index (0–1)")
    axes[0].axvline(median_risk, color="white", linestyle="--", alpha=0.5, linewidth=1)

    bottom = np.zeros(len(df))
    for col, label, color in [
        ("financial_pii_rate", "Financial PII", ACCENT),
        ("contact_pii_rate",   "Contact PII",   ACCENT2),
        ("identity_pii_rate",  "Identity PII",  ACCENT3),
    ]:
        vals = df[col].values if col in df.columns else np.zeros(len(df))
        axes[1].barh(df["genre_name"], vals, left=bottom,
                     label=label, color=color, edgecolor="#21262D",
                     linewidth=0.3, alpha=0.85)
        bottom += vals
    axes[1].set_title("PII Type Exposure Rate by Genre", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Proportion of Comments with PII")
    axes[1].legend(loc="lower right", fontsize=8)

    if "avg_sentiment" in df.columns:
        sent_colors = [ACCENT2 if v >= 0 else ACCENT for v in df["avg_sentiment"]]
        axes[2].barh(df["genre_name"], df["avg_sentiment"],
                     color=sent_colors, edgecolor="#21262D", linewidth=0.5)
        axes[2].axvline(0, color="white", linestyle="--", alpha=0.5, linewidth=1)
    axes[2].set_title("Average Comment Sentiment", fontweight="bold", fontsize=11)
    axes[2].set_xlabel("Sentiment Score (−1 to +1)")

    fig.suptitle("PII Profile by Content Genre",
                 fontsize=16, fontweight="bold", y=1.02, color=FG)
    plt.tight_layout()
    return fig


def fig_genre_age_risk(matrix: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, max(7, len(matrix) * 0.5)))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, linecolor="#21262D", ax=ax,
        cbar_kws={"label": "Privacy Susceptibility Score"},
        annot_kws={"size": 8},
    )
    ax.set_title(
        "Genre × Age Group — Privacy Susceptibility Matrix\n"
        "Higher score = age group is more at-risk within that content genre",
        fontsize=13, fontweight="bold", pad=14, color=FG,
    )
    ax.set_xlabel("Age Group", fontsize=10)
    ax.set_ylabel("Content Genre", fontsize=10)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig


def fig_top_risk_pairs(matrix: pd.DataFrame, n: int = 10) -> plt.Figure:
    stacked = matrix.stack().reset_index()
    stacked.columns = ["Genre", "Age Group", "Risk Score"]
    top = stacked.nlargest(n, "Risk Score")
    top["Label"] = top["Genre"] + "\n" + top["Age Group"]

    fig, ax = plt.subplots(figsize=(11, 5))
    palette = sns.color_palette("YlOrRd", len(top))[::-1]
    bars = ax.barh(top["Label"], top["Risk Score"],
                   color=palette, edgecolor="#21262D", linewidth=0.5)
    for bar, score in zip(bars, top["Risk Score"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9, color=FG)
    ax.set_title(f"Top {n} Highest Privacy Risk (Genre, Age Group) Pairs",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Privacy Susceptibility Score")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def fig_age_risk_bar(survey_df: pd.DataFrame) -> plt.Figure:
    AGE_ORDER = ["18 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65 or older"]
    if "age_enc" not in survey_df.columns or "risk_label" not in survey_df.columns:
        return None
    age_map = {i: v for i, v in enumerate(AGE_ORDER)}
    age_risk = survey_df.groupby("age_enc")["risk_label"].mean()
    age_risk.index = age_risk.index.map(age_map)
    age_risk = age_risk.dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    palette = sns.color_palette("magma", len(age_risk))
    ax.bar(range(len(age_risk)), age_risk.values,
           color=palette, edgecolor="#21262D", linewidth=0.4, width=0.65)
    ax.set_xticks(range(len(age_risk)))
    ax.set_xticklabels(age_risk.index, rotation=30, ha="right", fontsize=9)
    ax.axhline(age_risk.mean(), color=ACCENT, linestyle="--",
               linewidth=1.2, label=f"Mean: {age_risk.mean():.1%}")
    ax.set_title("High-Risk Rate by Age Group", fontweight="bold", fontsize=12)
    ax.set_ylabel("Proportion Classified High-Risk")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def fig_concern_scatter(survey_df: pd.DataFrame) -> plt.Figure:
    needed = ["concern_index", "behaviour_gap", "age_enc"]
    if not all(c in survey_df.columns for c in needed):
        return None
    sample = survey_df.sample(min(500, len(survey_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        sample["concern_index"], sample["behaviour_gap"],
        c=sample["age_enc"], cmap="plasma",
        alpha=0.55, s=22, edgecolors="none",
    )
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(sample["concern_index"].mean(), color=ACCENT2,
               linewidth=0.8, linestyle="--", alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Age Enc (0=18–24 → 5=65+)", fontsize=8, color=FG)
    cbar.ax.yaxis.set_tick_params(color=FG)
    ax.set_xlabel("Concern Index", fontsize=10)
    ax.set_ylabel("Behaviour Gap\n(Concern − Action)", fontsize=10)
    ax.set_title("Privacy Concern vs Behaviour Gap\n(coloured by age group)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


def fig_modality_pie(weights: dict) -> plt.Figure:
    if not weights:
        return None
    labels = list(weights.keys())
    vals   = list(weights.values())
    colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops={"edgecolor": BG, "linewidth": 2},
        textprops={"color": FG, "fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(BG)
        at.set_fontweight("bold")
    ax.set_title("Multimodal Fusion\nAttention Weights",
                 fontweight="bold", fontsize=12, color=FG)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(has_cache: bool):
    with st.sidebar:
        st.markdown("## 🔐 Privacy Study")
        st.markdown("---")

        st.markdown("### ⚙️ Configuration")

        supabase_url = st.text_input(
            "Supabase URL",
            value=os.getenv("SUPABASE_URL", ""),
            type="default",
            placeholder="https://xxxx.supabase.co",
            help="Your Supabase project URL"
        )
        supabase_key = st.text_input(
            "Supabase Key",
            value=os.getenv("SUPABASE_KEY", ""),
            type="password",
            placeholder="your-anon-key",
            help="Your Supabase anon/service key"
        )

        st.markdown("### 🧩 Modalities")
        run_pii   = st.toggle("PII Scanning (Presidio)", value=True,
                               help="Extract PII entities from comments")
        run_image = st.toggle("Image Analysis (OpenCV)", value=False,
                               help="Analyse thumbnail images (slow)")

        st.markdown("---")

        # Cache status
        if has_cache:
            st.markdown(
                '<span class="badge badge-green">● CACHE EXISTS</span>',
                unsafe_allow_html=True
            )
            try:
                with open(CACHE_DIR / "meta.json") as f:
                    meta = json.load(f)
                st.caption(f"Generated: {meta.get('generated_at', 'unknown')[:19]}")
                st.caption(f"Survey rows: {meta.get('n_survey', '—')}")
                st.caption(f"Genres: {meta.get('n_genres', '—')}")
            except Exception:
                pass
        else:
            st.markdown(
                '<span class="badge badge-red">● NO CACHE</span>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button("▶ Run Pipeline", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑 Clear Cache", use_container_width=True)

        if clear_btn:
            clear_cache()
            st.rerun()

        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        <small style="color:#8B949E; line-height:1.6">
        Multimodal AI pipeline for social media privacy research.<br><br>
        <b>Modalities:</b><br>
        ① Text — PII + Sentiment<br>
        ② Metadata — Engagement<br>
        ③ Image — Thumbnails<br>
        ④ Survey — Behavioural<br><br>
        <b>Outputs:</b><br>
        • Genre × Age affinity<br>
        • PII risk profiles<br>
        • Susceptibility matrix<br>
        • GB classifier metrics
        </small>
        """, unsafe_allow_html=True)

    return supabase_url, supabase_key, run_pii, run_image, run_btn


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(data: dict):
    affinity  = data.get("genre_age_affinity",  pd.DataFrame())
    pii_prof  = data.get("genre_pii_profile",   pd.DataFrame())
    risk_mat  = data.get("genre_age_risk",       pd.DataFrame())
    survey    = data.get("survey",               pd.DataFrame())
    metrics   = data.get("metrics",              {})
    mod_w     = data.get("modality_weights",     {})
    meta      = data.get("meta",                 {})

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div style="font-size:11px; letter-spacing:0.15em; color:#8B949E; 
                  font-family:'JetBrains Mono',monospace; margin-bottom:8px">
        PRIVACY PERCEPTION STUDY  ·  MULTIMODAL ANALYSIS
      </div>
      <h1 style="margin:0 0 8px 0; font-size:2.2rem; letter-spacing:-0.03em">
        Genre × Age × PII Dashboard
      </h1>
      <p style="color:#8B949E; margin:0; font-size:0.95rem">
        Correlating content genres, age demographics, and personal information exposure 
        across social media platforms.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_survey = len(survey) if not survey.empty else meta.get("n_survey", "—")
        st.metric("Survey Respondents", f"{n_survey:,}" if isinstance(n_survey, int) else n_survey)
    with c2:
        n_genres = len(pii_prof) if not pii_prof.empty else meta.get("n_genres", "—")
        st.metric("Content Genres", n_genres)
    with c3:
        auc = metrics.get("auc_roc", None)
        st.metric("Model AUC-ROC", f"{auc:.3f}" if auc else "—")
    with c4:
        f1 = metrics.get("f1", None)
        st.metric("Model F1 Score", f"{f1:.3f}" if f1 else "—")
    with c5:
        if not survey.empty and "risk_label" in survey.columns:
            hr = survey["risk_label"].mean()
            st.metric("High-Risk Rate", f"{hr:.1%}")
        else:
            st.metric("High-Risk Rate", "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Genre–Age Affinity",
        "🔍 PII Profiles",
        "🌡️ Risk Matrix",
        "🏆 Top Risk Pairs",
        "👤 Demographics",
        "🤖 Model Insights",
        "📋 Raw Data",
    ])

    # ── TAB 1: Genre–Age Affinity ─────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Genre–Age Affinity Matrix")
        st.caption("Each row shows the proportion of that genre's audience by age group — "
                   "P(age | genre consumed). Darker blue = stronger affinity.")
        if not affinity.empty:
            col_filter, col_sort = st.columns([3, 1])
            with col_filter:
                all_genres = list(affinity.index)
                sel_genres = st.multiselect("Filter genres", all_genres,
                                            default=all_genres,
                                            key="aff_genre_filter")
            with col_sort:
                sort_by = st.selectbox("Sort by age group", ["—"] + list(affinity.columns),
                                       key="aff_sort")

            sub = affinity.loc[sel_genres] if sel_genres else affinity
            if sort_by != "—" and sort_by in sub.columns:
                sub = sub.sort_values(sort_by, ascending=False)

            fig = fig_genre_age_affinity(sub)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📌 Key Observations")
            c1, c2 = st.columns(2)
            with c1:
                if "18 - 24" in affinity.columns:
                    top_youth = affinity["18 - 24"].nlargest(3)
                    st.markdown("**Top genres for 18–24:**")
                    for g, v in top_youth.items():
                        st.markdown(f"- **{g}** — {v:.0%}")
            with c2:
                oldest_cols = [c for c in ["45 - 54", "55 - 64", "65 or older"] if c in affinity.columns]
                if oldest_cols:
                    top_old = affinity[oldest_cols].sum(axis=1).nlargest(3)
                    st.markdown("**Most age-diverse genres (45+):**")
                    for g, v in top_old.items():
                        st.markdown(f"- **{g}** — {v:.0%} from 45+")
        else:
            st.info("No affinity data available. Run the pipeline first.")

    # ── TAB 2: PII Profiles ───────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### PII Profile by Content Genre")
        st.caption("Risk index, PII type breakdown, and sentiment score per genre.")
        if not pii_prof.empty:
            fig = fig_genre_pii_profile(pii_prof)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔎 Genre Detail Explorer")
            genre_sel = st.selectbox("Select a genre", pii_prof["genre_name"].tolist(),
                                     key="pii_genre_sel")
            row = pii_prof[pii_prof["genre_name"] == genre_sel].iloc[0]
            cols = st.columns(4)
            kpis = [
                ("Risk Index",     f"{row['genre_risk_index']:.3f}", ""),
                ("PII Rate",       f"{row['pii_rate']:.1%}",         "comments with any PII"),
                ("Avg Sentiment",  f"{row['avg_sentiment']:.3f}",    "−1 negative → +1 positive"),
                ("Comments",       f"{int(row.get('n_comments', 0)):,}", "analysed"),
            ]
            for col, (label, val, help_) in zip(cols, kpis):
                col.metric(label, val, delta=help_ if help_ else None,
                           delta_color="off" if help_ else "normal")

            pii_row = st.columns(3)
            pii_types = [
                ("💳 Financial PII", row.get("financial_pii_rate", 0)),
                ("📧 Contact PII",   row.get("contact_pii_rate",   0)),
                ("🆔 Identity PII",  row.get("identity_pii_rate",  0)),
            ]
            for col, (label, val) in zip(pii_row, pii_types):
                col.metric(label, f"{val:.1%}")
        else:
            st.info("No PII profile data. Run the pipeline first.")

    # ── TAB 3: Risk Matrix ────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Privacy Susceptibility Matrix")
        st.caption("Combined score from genre risk × age demographic × behavioural signals. "
                   "Higher = more at-risk.")
        if not risk_mat.empty:
            n_show = st.slider("Genres to display", 5, len(risk_mat), min(20, len(risk_mat)),
                               key="risk_n_show")
            # Sort by max risk across age groups
            top_genres = risk_mat.max(axis=1).nlargest(n_show).index
            sub = risk_mat.loc[top_genres]
            fig = fig_genre_age_risk(sub)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("<br>", unsafe_allow_html=True)
            # Summary stats
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Highest single-cell risk:**")
                stacked = risk_mat.stack()
                top_cell = stacked.idxmax()
                st.markdown(f"🔴 **{top_cell[0]}** × **{top_cell[1]}** "
                            f"= `{stacked.max():.3f}`")
            with c2:
                st.markdown("**Most at-risk age group overall:**")
                col_mean = risk_mat.mean(axis=0).idxmax()
                st.markdown(f"🎯 **{col_mean}** "
                            f"(avg `{risk_mat.mean(axis=0).max():.3f}`)")
            with c3:
                st.markdown("**Highest-risk genre overall:**")
                row_mean = risk_mat.mean(axis=1).idxmax()
                st.markdown(f"⚠️ **{row_mean}** "
                            f"(avg `{risk_mat.mean(axis=1).max():.3f}`)")
        else:
            st.info("No risk matrix data. Run the pipeline first.")

    # ── TAB 4: Top Risk Pairs ─────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Top Privacy Risk Pairs")
        st.caption("The (genre, age group) combinations with the highest susceptibility scores.")
        if not risk_mat.empty:
            n_top = st.slider("Number of pairs", 5, 20, 10, key="top_n")
            fig = fig_top_risk_pairs(risk_mat, n=n_top)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            stacked = risk_mat.stack().reset_index()
            stacked.columns = ["Genre", "Age Group", "Risk Score"]
            top_df = stacked.nlargest(n_top, "Risk Score").reset_index(drop=True)
            top_df.index += 1
            top_df["Risk Score"] = top_df["Risk Score"].map(lambda x: f"{x:.4f}")
            st.dataframe(top_df, use_container_width=True)
        else:
            st.info("No risk matrix data. Run the pipeline first.")

    # ── TAB 5: Demographics ───────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### Demographic Analysis")
        if not survey.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = fig_age_risk_bar(survey)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("Age risk data not available.")
            with c2:
                fig = fig_concern_scatter(survey)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("Concern/behaviour gap data not available.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Survey Summary Statistics")
            likert_cols = [c for c in [
                "q12_concern_data_use", "q13_trust_platforms",
                "q14_perceived_control", "q15_reviews_privacy_settings",
                "concern_index", "behaviour_gap",
            ] if c in survey.columns]
            if likert_cols:
                desc = survey[likert_cols].describe().T
                desc = desc.round(3)
                st.dataframe(desc, use_container_width=True)
        else:
            st.info("No survey data. Run the pipeline first.")

    # ── TAB 6: Model Insights ─────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### Model Performance & Interpretation")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Classification Metrics")
            if metrics:
                m_cols = st.columns(4)
                for col, (k, label) in zip(m_cols, [
                    ("auc_roc",   "AUC-ROC"),
                    ("precision", "Precision"),
                    ("recall",    "Recall"),
                    ("f1",        "F1 Score"),
                ]):
                    val = metrics.get(k)
                    col.metric(label, f"{val:.4f}" if val is not None else "—")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div class="section-card">
                  <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#8B949E; 
                              line-height:2">
                    <b style="color:#4ECDC4">Model</b> : Gradient Boosting Classifier<br>
                    <b style="color:#4ECDC4">CV</b>     : 5-Fold Stratified<br>
                    <b style="color:#4ECDC4">Label</b>  : Behavioural risk score ≥ 3/5 → High-Risk<br>
                    <b style="color:#4ECDC4">Split</b>  : 80% train / 20% test
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No model metrics. Run the pipeline first.")
    # ── TAB 7: Raw Data ───────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("### Raw Data Explorer")
        dataset = st.selectbox("Dataset", [
            "Genre–Age Affinity Matrix",
            "Genre PII Profile",
            "Genre × Age Risk Matrix",
            "Survey (sample)",
        ], key="raw_data_sel")

        mapping = {
            "Genre–Age Affinity Matrix": affinity,
            "Genre PII Profile":         pii_prof,
            "Genre × Age Risk Matrix":   risk_mat,
            "Survey (sample)":           survey.sample(min(200, len(survey)), random_state=42)
                                         if not survey.empty else pd.DataFrame(),
        }
        df_show = mapping[dataset]
        if not df_show.empty:
            st.dataframe(df_show.round(4), use_container_width=True, height=420)
            csv = df_show.to_csv().encode("utf-8")
            st.download_button(
                f"⬇ Download {dataset} as CSV",
                data=csv,
                file_name=f"{dataset.lower().replace(' ', '_').replace('×','x')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No data available for this dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE UI
# ─────────────────────────────────────────────────────────────────────────────

def render_run_pipeline_ui(supabase_url: str, supabase_key: str,
                            run_pii: bool, run_image: bool):
    st.markdown("""
    <div class="hero" style="border-color:#FF6B6B33">
      <div style="font-size:11px; letter-spacing:0.15em; color:#8B949E; 
                  font-family:'JetBrains Mono',monospace; margin-bottom:8px">
        NO CACHED DATA FOUND
      </div>
      <h2 style="margin:0 0 8px 0; color:#FF6B6B">Pipeline Not Yet Run</h2>
      <p style="color:#8B949E; margin:0">
        Configure your Supabase credentials in the sidebar and click <b>▶ Run Pipeline</b> 
        to fetch data, run analysis, and cache results.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### What the pipeline does:")
    steps = [
        ("1️⃣", "Load", "Fetches survey, comments, videos, genres from Supabase"),
        ("2️⃣", "Preprocess", "Cleans, validates, and encodes all data"),
        ("3️⃣", "Extract", "PII detection (Presidio) + sentiment (HuggingFace)"),
        ("4️⃣", "Fuse", "Hybrid attention fusion across modalities"),
        ("5️⃣", "Analyse", "Builds affinity, PII profile, and risk matrices"),
        ("6️⃣", "Model", "Trains Gradient Boosting classifier with 5-fold CV"),
        ("7️⃣", "Cache", "Saves all results to ./pipeline_cache/"),
    ]
    for icon, title, desc in steps:
        st.markdown(
            f'<div class="section-card" style="padding:12px 20px; margin-bottom:8px">'
            f'<span style="font-size:1.1em">{icon}</span> '
            f'<b style="color:{ACCENT2}">{title}</b> '
            f'<span style="color:#8B949E; font-size:0.9em">— {desc}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    if not supabase_url or not supabase_key or \
       supabase_url == "YOUR_SUPABASE_URL":
        st.warning("⚠️ Please enter valid Supabase credentials in the sidebar before running.")


# ─────────────────────────────────────────────────────────────────────────────
# APP ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    has_cache = cache_exists()

    supabase_url, supabase_key, run_pii, run_image, run_btn = render_sidebar(has_cache)

    # ── Handle Run Pipeline ───────────────────────────────────────────────────
    if run_btn:
        if not supabase_url or not supabase_key:
            st.error("❌ Please provide Supabase URL and Key in the sidebar.")
            st.stop()

        st.markdown("## ⚙️ Running Pipeline …")
        log_placeholder = st.empty()
        progress = st.progress(0, text="Initialising …")

        try:
            progress.progress(10, "Connecting to Supabase …")
            results = run_pipeline(supabase_url, supabase_key,
                                   run_pii, run_image, log_placeholder)
            progress.progress(80, "Saving cache …")
            save_cache(results)
            progress.progress(100, "Done!")
            st.session_state["pipeline_results"] = results
            st.success("✅ Pipeline complete! Reloading dashboard …")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
            st.stop()

    # ── Load or show prompt ───────────────────────────────────────────────────
    if "pipeline_results" in st.session_state:
        data = st.session_state["pipeline_results"]
        render_dashboard(data)
    elif has_cache:
        data = load_cache()
        render_dashboard(data)
    else:
        render_run_pipeline_ui(supabase_url, supabase_key, run_pii, run_image)


if __name__ == "__main__":
    main()
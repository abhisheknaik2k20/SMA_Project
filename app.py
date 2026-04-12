"""
app.py
──────
Streamlit dashboard for the Privacy Perception Study.

On first launch  → runs full pipeline (Supabase fetch + compute) → saves to disk
On every later launch → loads saved results from disk, NO DB/compute work done
"""

import os
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from cache_manager import cache_exists, load_cache, clear_cache, cache_info

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Privacy Perception Study",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0D1117; }
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    .metric-card {
        background: #161B22; border: 1px solid #30363D;
        border-radius: 10px; padding: 1rem 1.2rem;
        text-align: center; margin-bottom: 0.4rem;
    }
    .metric-card .label {
        font-size: 0.72rem; color: #8B949E;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem;
    }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; color: #E6EDF3; }
    .metric-card .delta { font-size: 0.78rem; color: #4ECDC4; margin-top: 0.15rem; }
    .section-header {
        background: linear-gradient(90deg,#FF6B6B22 0%,transparent 100%);
        border-left: 3px solid #FF6B6B; padding: 0.45rem 0.9rem;
        border-radius: 0 6px 6px 0; margin: 1rem 0 0.7rem 0;
        color: #E6EDF3; font-weight: 600; font-size: 1rem;
    }
    .cache-badge {
        display:inline-block; background:#238636; color:#fff;
        border-radius:12px; padding:2px 12px; font-size:0.75rem; font-weight:600;
    }
    .nocache-badge {
        display:inline-block; background:#DA3633; color:#fff;
        border-radius:12px; padding:2px 12px; font-size:0.75rem; font-weight:600;
    }
    section[data-testid="stSidebar"] { background-color: #161B22; }
    .stTabs [data-baseweb="tab-list"] { background-color:#161B22; border-radius:8px; }
    .stTabs [data-baseweb="tab"] { color:#8B949E; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color:#E6EDF3; }
    .log-box {
        background:#0D1117; border:1px solid #30363D; border-radius:8px;
        padding:1rem; font-family:monospace; font-size:0.78rem;
        color:#4ECDC4; max-height:320px; overflow-y:auto; white-space:pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

AGE_ORDER = ["18 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65 or older"]
GENRE_RISK_PRIOR = {
    "News & Politics": 0.75, "Education": 0.45, "Entertainment": 0.55,
    "Gaming": 0.50, "Music": 0.40, "Science & Technology": 0.60,
    "Health": 0.70, "Finance": 0.80, "Travel": 0.65, "Food": 0.35,
    "Sports": 0.45, "Fashion & Beauty": 0.50, "Lifestyle": 0.55,
    "Comedy": 0.40, "Other": 0.50,
}
BG   = "#0D1117"
CARD = "#161B22"
RED  = "#FF6B6B"
TEAL = "#4ECDC4"
YELL = "#FFE66D"
PURP = "#C77DFF"
TMPL = "plotly_dark"

NEEDED_KEYS = ["survey", "comments", "affinity_mat",
               "genre_pii", "risk_matrix", "model_results"]

def all_cached() -> bool:
    return all(cache_exists(k) for k in NEEDED_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FROM DISK  (one Streamlit resource cache per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all_from_cache():
    return {
        "survey":        load_cache("survey"),
        "comments":      load_cache("comments"),
        "affinity_mat":  load_cache("affinity_mat"),
        "affinity_df":   load_cache("affinity_df"),
        "genre_pii":     load_cache("genre_pii"),
        "risk_matrix":   load_cache("risk_matrix"),
        "model_results": load_cache("model_results") or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(cached: bool) -> dict:
    st.sidebar.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem 0;">
      <div style="font-size:2rem;">🔐</div>
      <div style="font-size:1.05rem;font-weight:700;color:#E6EDF3;">Privacy Study</div>
      <div style="font-size:0.7rem;color:#8B949E;">Multimodal Analysis Pipeline</div>
    </div><hr style="border-color:#30363D;margin:0.6rem 0;">
    """, unsafe_allow_html=True)

    badge = ('<span class="cache-badge">✓ Loaded from disk</span>' if cached
             else '<span class="nocache-badge">No cache — pipeline required</span>')
    st.sidebar.markdown(f"**Status:** {badge}", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🗄️ Supabase Credentials")
    st.sidebar.caption("Only required on first run. Results are then saved to disk.")

    sb_url = st.sidebar.text_input("Supabase URL",
        value=os.getenv("SUPABASE_URL", ""),
        placeholder="https://xxx.supabase.co")
    sb_key = st.sidebar.text_input("Supabase Anon Key",
        value=os.getenv("SUPABASE_KEY", ""),
        type="password", placeholder="eyJ…")

    run_btn = st.sidebar.button(
        "▶ Fetch from Supabase & Run Pipeline",
        use_container_width=True, type="primary",
        disabled=not (sb_url and sb_key),
        help="Runs once — saves everything to disk.",
    )

    st.sidebar.markdown("---")
    clear_btn = st.sidebar.button(
        "🗑️ Clear Disk Cache",
        use_container_width=True, type="secondary",
        help="Deletes saved results. Next click of Run will re-run.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎛️ Display Filters")
    age_filter   = st.sidebar.multiselect("Age Groups", AGE_ORDER, default=AGE_ORDER)
    genre_filter = st.sidebar.multiselect("Genres", list(GENRE_RISK_PRIOR.keys()),
                                           default=list(GENRE_RISK_PRIOR.keys()))
    top_n        = st.sidebar.slider("Top-N risk pairs", 5, 20, 10)

    if cached:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📦 Cached Artefacts")
        for k, v in cache_info().items():
            st.sidebar.markdown(
                f"<div style='font-size:0.7rem;color:#8B949E;line-height:1.6;'>"
                f"<b>{k}</b> · {v['size_kb']} KB<br>"
                f"<span style='color:#484F58;'>{v.get('saved_at','')[:19]}</span></div>",
                unsafe_allow_html=True)

    return dict(sb_url=sb_url, sb_key=sb_key,
                run_btn=run_btn, clear_btn=clear_btn,
                age_filter=age_filter, genre_filter=genre_filter, top_n=top_n)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER  (live log in main area)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_ui(sb_url: str, sb_key: str):
    from pipeline import run_full_pipeline

    st.markdown('<div class="section-header">🚀 Running Pipeline (first time only)</div>',
                unsafe_allow_html=True)
    placeholder = st.empty()
    lines: list = []

    def log(msg: str):
        lines.append(msg)
        placeholder.markdown(
            "<div class='log-box'>" + "\n".join(lines[-40:]) + "</div>",
            unsafe_allow_html=True)

    with st.spinner("Pipeline running — this is a one-time operation…"):
        ok = run_full_pipeline(sb_url, sb_key, log)

    if ok:
        st.success("✅ Done! Results saved to disk. Reloading dashboard…")
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error("Pipeline failed — check the log above.")


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _L(fig, title="", h=420, **kw):
    fig.update_layout(
        template=TMPL, paper_bgcolor=BG, plot_bgcolor=CARD,
        height=h, font=dict(color="#E6EDF3"),
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=40, r=30, t=50, b=50), **kw)
    return fig

def chart_risk_heatmap(rm):
    fig = px.imshow(rm, color_continuous_scale="YlOrRd", aspect="auto",
                    text_auto=".3f", zmin=0, zmax=1,
                    labels={"color":"Susceptibility"},
                    title="Genre × Age — Privacy Susceptibility")
    fig.update_xaxes(tickangle=-30)
    return _L(fig, h=480)

def chart_top_pairs(rm, n):
    s = rm.stack().reset_index()
    s.columns = ["Genre","Age","Score"]
    top = s.nlargest(n,"Score").sort_values("Score")
    top["Label"] = top["Genre"] + " · " + top["Age"]
    fig = go.Figure(go.Bar(
        y=top["Label"], x=top["Score"], orientation="h",
        marker=dict(color=top["Score"], colorscale="YlOrRd", showscale=True),
        text=top["Score"].round(3), textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
    ))
    return _L(fig, f"Top {n} Risk Pairs", h=420, xaxis_title="Susceptibility Score")

def chart_age_risk(df):
    ar = df.groupby("age_enc")["risk_label"].mean().reset_index()
    ar["age_label"] = ar["age_enc"].map({i:v for i,v in enumerate(AGE_ORDER)})
    ar = ar.dropna()
    fig = go.Figure(go.Bar(
        x=ar["age_label"], y=ar["risk_label"],
        marker=dict(color=ar["risk_label"], colorscale="Magma"),
        text=(ar["risk_label"]*100).round(1).astype(str)+"%",
        textposition="outside"))
    fig.add_hline(y=ar["risk_label"].mean(), line_dash="dash",
                  line_color=RED, opacity=0.7, annotation_text="avg")
    fig.update_xaxes(tickangle=-30)
    return _L(fig, "High-Risk Rate by Age Group", h=360, yaxis_title="Proportion High-Risk")

def chart_genre_risk(gp):
    df = gp.sort_values("genre_risk_index")
    med = df["genre_risk_index"].median()
    fig = go.Figure(go.Bar(
        y=df["genre_name"], x=df["genre_risk_index"], orientation="h",
        marker_color=[RED if v > med else TEAL for v in df["genre_risk_index"]],
        text=df["genre_risk_index"].round(3), textposition="outside",
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=med, line_dash="dash", line_color="white", opacity=0.5,
                  annotation_text="median")
    return _L(fig, "Composite Genre Risk Index", h=430, xaxis_title="Risk Index (0–1)")

def chart_affinity(mat):
    fig = px.imshow(mat*100, color_continuous_scale="Blues", aspect="auto",
                    text_auto=".0f", labels={"color":"% audience"},
                    title="Genre–Age Affinity Matrix  |  P(age | genre)")
    fig.update_xaxes(tickangle=-30)
    return _L(fig, h=430)

def chart_pii_stacked(gp):
    df = gp.sort_values("genre_risk_index")
    fig = go.Figure()
    for col, label, color in [
        ("financial_pii_rate","Financial PII",RED),
        ("contact_pii_rate","Contact PII",TEAL),
        ("identity_pii_rate","Identity PII",YELL)]:
        fig.add_trace(go.Bar(
            y=df["genre_name"], x=df.get(col,0),
            orientation="h", name=label, marker_color=color, opacity=0.85))
    fig.update_layout(barmode="stack")
    return _L(fig,"PII Type Exposure Rate by Genre",h=430,
              xaxis_title="Proportion with PII",
              legend=dict(orientation="h",y=1.08))

def chart_sentiment(gp):
    df = gp.sort_values("avg_sentiment")
    fig = go.Figure(go.Bar(
        y=df["genre_name"], x=df["avg_sentiment"], orientation="h",
        marker_color=[TEAL if v>=0 else RED for v in df["avg_sentiment"]]))
    fig.add_vline(x=0, line_dash="solid", line_color="white", opacity=0.5)
    return _L(fig,"Avg Comment Sentiment by Genre",h=430,xaxis_title="Sentiment (−1 to +1)")

def chart_entity_by_genre(cdf):
    rows = []
    for _, row in cdf.iterrows():
        for et in (row.get("pii_entity_types") or []):
            rows.append({"genre_name":row["genre_name"],"entity_type":et})
    if not rows:
        return go.Figure()
    edf = pd.DataFrame(rows)
    top = edf["entity_type"].value_counts().head(6).index.tolist()
    edf = edf[edf["entity_type"].isin(top)]
    piv = edf.groupby(["genre_name","entity_type"]).size().reset_index(name="count")
    fig = px.bar(piv, x="genre_name", y="count", color="entity_type",
                 barmode="group",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title="PII Entity Distribution by Genre",
                 labels={"genre_name":"Genre","count":"Count","entity_type":"PII Type"})
    fig.update_xaxes(tickangle=-30)
    return _L(fig,h=400,legend=dict(orientation="h",y=1.08))

def chart_radar(row, genre):
    cats = ["Financial PII","Contact PII","Identity PII","Avg Sensitivity","Risk Index"]
    vals = [row.get("financial_pii_rate",0), row.get("contact_pii_rate",0),
            row.get("identity_pii_rate",0), row.get("avg_sensitivity",0),
            row.get("genre_risk_index",0)]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor=RED+"33", line_color=RED, name=genre))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,1],color="#8B949E"),bgcolor=CARD),
        showlegend=False, template=TMPL, paper_bgcolor=BG, height=300,
        title=f"PII Profile — {genre}", font=dict(color="#E6EDF3"),
        margin=dict(l=40,r=40,t=50,b=40))
    return fig

def chart_feat_importance(fi_dict):
    fi = pd.Series(fi_dict).sort_values()
    fig = go.Figure(go.Bar(
        y=fi.index, x=fi.values, orientation="h",
        marker_color=[RED if i>=len(fi)-3 else PURP for i in range(len(fi))],
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
    return _L(fig,"Feature Importance",h=430,xaxis_title="Mean Decrease in Impurity")

def chart_auc_gauge(auc):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=auc,
        title={"text":"AUC-ROC","font":{"color":"#E6EDF3"}},
        delta={"reference":0.7,"increasing":{"color":TEAL}},
        gauge={"axis":{"range":[0.5,1.0],"tickcolor":"#8B949E"},
               "bar":{"color":TEAL},"bgcolor":CARD,"bordercolor":"#30363D",
               "steps":[{"range":[0.5,0.65],"color":RED+"44"},
                        {"range":[0.65,0.8],"color":YELL+"44"},
                        {"range":[0.8,1.0],"color":TEAL+"44"}],
               "threshold":{"line":{"color":"white","width":2},"value":0.8}},
        number={"font":{"color":"#E6EDF3"}}))
    fig.update_layout(paper_bgcolor=BG,font={"color":"#E6EDF3"},
                      height=270,margin=dict(l=30,r=30,t=50,b=10))
    return fig

def chart_concern_scatter(df):
    s = df.sample(min(400,len(df)),random_state=42)
    fig = px.scatter(s, x="concern_index", y="behaviour_gap",
                     color="age_enc", color_continuous_scale="Plasma", opacity=0.55,
                     hover_data={"q1_age_range":True,"risk_label":True,"age_enc":False},
                     title="Concern vs Behaviour Gap")
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    return _L(fig,h=370)

def chart_platform(df):
    if "platforms" not in df.columns: return go.Figure()
    ps = df["platforms"].explode()
    ps = ps[ps.notna() & (ps!="nan")]
    c = ps.value_counts().head(10)
    fig = go.Figure(go.Bar(x=c.index, y=c.values,
        marker=dict(color=c.values, colorscale="Teal"),
        text=c.values, textposition="outside"))
    return _L(fig,"Platform Usage",h=350,yaxis_title="# Respondents")

def chart_behaviour_score(df):
    if "behaviour_score" not in df.columns: return go.Figure()
    bs = df["behaviour_score"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[str(int(v)) for v in bs.index], y=bs.values,
        marker_color=[TEAL if v<3 else RED for v in bs.index],
        text=bs.values, textposition="outside"))
    fig.add_vline(x=2.5,line_dash="dash",line_color="white",annotation_text="Risk threshold")
    return _L(fig,"Behaviour Score Distribution  (≥3 = High Risk)",h=320,
              xaxis_title="Risky Behaviours (0–5)",yaxis_title="Respondents")


def mcard(label, value, delta=None):
    d = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{d}</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cached = all_cached()
    opts   = render_sidebar(cached)

    # Sidebar actions
    if opts["clear_btn"]:
        clear_cache()
        st.cache_resource.clear()
        st.rerun()

    if opts["run_btn"]:
        run_pipeline_ui(opts["sb_url"], opts["sb_key"])
        st.stop()

    # Header
    st.markdown("""
    <div style="padding:0.3rem 0 1rem 0;">
        <h1 style="margin:0;font-size:1.7rem;color:#E6EDF3;">🔐 Privacy Perception Study</h1>
        <p style="color:#8B949E;margin:0.2rem 0 0 0;font-size:0.88rem;">
            Multimodal Analysis — Genre × Age × PII Privacy Susceptibility
        </p>
    </div>""", unsafe_allow_html=True)

    # No cache state
    if not cached:
        st.warning("No results on disk yet. Enter Supabase credentials in the sidebar "
                   "and click **▶ Fetch from Supabase & Run Pipeline**.", icon="⚠️")
        st.markdown("""
        <div style="background:#161B22;border:1px solid #30363D;border-radius:10px;
                    padding:1.5rem 2rem;max-width:620px;margin-top:1rem;">
          <h3 style="color:#E6EDF3;margin-top:0;">How it works</h3>
          <ol style="color:#8B949E;line-height:2.2;">
            <li>Enter your <b style="color:#E6EDF3;">Supabase URL + Key</b> in the sidebar</li>
            <li>Click <b style="color:#4ECDC4;">▶ Fetch from Supabase & Run Pipeline</b></li>
            <li>The app fetches all tables, extracts PII, builds the Genre × Age
                risk matrix, and trains the model — <b style="color:#E6EDF3;">one time only</b></li>
            <li>Every result is <b style="color:#E6EDF3;">saved to disk</b>
                in <code style="color:#4ECDC4;">.pipeline_cache/</code></li>
            <li>Every future app launch <b style="color:#E6EDF3;">loads from disk instantly</b>
                — zero DB calls, zero recomputation</li>
          </ol>
          <div style="color:#484F58;font-size:0.8rem;margin-top:0.5rem;">
            Use <b>Clear Disk Cache</b> when you want to pull fresh data from Supabase.
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # Load from disk
    data         = load_all_from_cache()
    survey_df    = data["survey"]
    comments_df  = data["comments"]
    affinity_mat = data["affinity_mat"]
    genre_pii    = data["genre_pii"]
    risk_matrix  = data["risk_matrix"]
    model_res    = data["model_results"] or {}

    if survey_df is None or survey_df.empty:
        st.error("Cached data appears corrupted — clear the cache and re-run.")
        return

    # Filters
    af = opts["age_filter"]
    gf = opts["genre_filter"]

    s_df  = survey_df[survey_df["q1_age_range"].isin(af)]          if af else survey_df
    g_pii = genre_pii[genre_pii["genre_name"].isin(gf)]            if gf else genre_pii
    r_mat = risk_matrix.loc[
                risk_matrix.index.isin(gf),
                [c for c in risk_matrix.columns if c in af]]        if gf and af else risk_matrix
    c_df  = comments_df[comments_df["genre_name"].isin(gf)]        if gf else comments_df
    a_mat = affinity_mat.loc[
                affinity_mat.index.isin(gf),
                [c for c in affinity_mat.columns if c in af]]       if gf and af else affinity_mat

    # ── Key metrics ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
    mc = st.columns(6)
    with mc[0]: mcard("Respondents",   f"{len(survey_df):,}")
    with mc[1]: mcard("Comments",      f"{len(comments_df):,}")
    with mc[2]: mcard("High-Risk Rate",f"{survey_df['risk_label'].mean()*100:.1f}%")
    with mc[3]: mcard("Genres",        str(len(genre_pii)))
    with mc[4]: mcard("AUC-ROC",       str(model_res.get("auc_roc","—")))
    with mc[5]: mcard("F1 Score",      str(model_res.get("f1","—")))

    ci = cache_info()
    if "survey" in ci:
        st.caption(f"📦 All results loaded from disk · "
                   f"Computed at {ci['survey'].get('saved_at','')[:19]} · "
                   f"Clear cache in sidebar to refresh.")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🎯 Risk Overview","👥 Affinity","🔍 PII","📈 Model",
                    "🧑 Survey","📋 Explorer"])

    # TAB 0 – Risk Overview
    with tabs[0]:
        st.markdown('<div class="section-header">🗺️ Privacy Susceptibility</div>',
                    unsafe_allow_html=True)
        c1,c2 = st.columns([3,2])
        with c1: st.plotly_chart(chart_risk_heatmap(r_mat), use_container_width=True)
        with c2: st.plotly_chart(chart_top_pairs(r_mat, opts["top_n"]), use_container_width=True)
        c3,c4 = st.columns(2)
        with c3: st.plotly_chart(chart_age_risk(s_df),     use_container_width=True)
        with c4: st.plotly_chart(chart_genre_risk(g_pii),  use_container_width=True)
        if not r_mat.empty and r_mat.size:
            idx = r_mat.stack().idxmax()
            st.info(f"⚠️ **Highest risk pair:** **{idx[0]}** × **{idx[1]}** "
                    f"(score: {r_mat.stack().max():.3f})", icon="🔴")

    # TAB 1 – Affinity
    with tabs[1]:
        st.markdown('<div class="section-header">👥 Genre–Age Affinity</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_affinity(a_mat), use_container_width=True)
        st.markdown("#### 🎯 Drill into a genre")
        sel = st.selectbox("Genre:", a_mat.index.tolist() if not a_mat.empty else ["—"])
        if sel and sel != "—" and sel in a_mat.index:
            row = a_mat.loc[sel]
            co1,co2 = st.columns(2)
            with co1:
                fig_pie = go.Figure(go.Pie(
                    labels=row.index, values=row.values*100, hole=0.4,
                    marker_colors=px.colors.sequential.Blues[2:]))
                fig_pie.update_layout(template=TMPL,paper_bgcolor=BG,height=280,
                    title=f"Age breakdown — {sel}",font=dict(color="#E6EDF3"),
                    margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig_pie, use_container_width=True)
            with co2:
                st.markdown(f"**Affinity values for {sel}:**")
                st.dataframe(row.rename("Proportion").to_frame().style.format("{:.1%}"),
                             use_container_width=True)

    # TAB 2 – PII
    with tabs[2]:
        st.markdown('<div class="section-header">🔍 PII Exposure Analysis</div>',
                    unsafe_allow_html=True)
        co1,co2 = st.columns(2)
        with co1: st.plotly_chart(chart_pii_stacked(g_pii), use_container_width=True)
        with co2: st.plotly_chart(chart_sentiment(g_pii),   use_container_width=True)
        st.plotly_chart(chart_entity_by_genre(c_df), use_container_width=True)
        st.markdown("#### 🔬 Genre PII Radar")
        sel_g = st.selectbox("Genre for radar:", g_pii["genre_name"].tolist(), key="radar")
        if sel_g:
            row = g_pii[g_pii["genre_name"]==sel_g].iloc[0]
            mc2 = st.columns(4)
            with mc2[0]: mcard("PII Rate",   f"{row['pii_rate']:.1%}")
            with mc2[1]: mcard("Avg PII",    f"{row['avg_pii_count']:.2f}")
            with mc2[2]: mcard("Risk Index", f"{row['genre_risk_index']:.3f}")
            with mc2[3]: mcard("Sentiment",  f"{row['avg_sentiment']:+.3f}")
            st.plotly_chart(chart_radar(row, sel_g), use_container_width=True)

    # TAB 3 – Model
    with tabs[3]:
        st.markdown('<div class="section-header">📈 Predictive Model</div>',
                    unsafe_allow_html=True)
        if not model_res:
            st.info("No model results in cache.")
        else:
            m = st.columns(4)
            with m[0]: mcard("AUC-ROC",   str(model_res.get("auc_roc","—")),
                             f"CV: {model_res.get('cv_auc_mean','—')} ±{model_res.get('cv_auc_std','—')}")
            with m[1]: mcard("Precision", str(model_res.get("precision","—")))
            with m[2]: mcard("Recall",    str(model_res.get("recall","—")))
            with m[3]: mcard("F1",        str(model_res.get("f1","—")),
                             f"CV: {model_res.get('cv_f1_mean','—')}")
            st.markdown("---")
            co1,co2 = st.columns([3,2])
            with co1:
                if model_res.get("feature_importances"):
                    st.plotly_chart(chart_feat_importance(model_res["feature_importances"]),
                                    use_container_width=True)
            with co2:
                st.markdown("#### Classification Report")
                st.code(model_res.get("class_report","—"), language="text")
                st.markdown(f"Train: `{model_res.get('n_train','—')}` · "
                            f"Test: `{model_res.get('n_test','—')}` · "
                            f"Gradient Boosting, 5-fold CV")
            if model_res.get("auc_roc"):
                st.plotly_chart(chart_auc_gauge(model_res["auc_roc"]),
                                use_container_width=True)

    # TAB 4 – Survey
    with tabs[4]:
        st.markdown('<div class="section-header">🧑 Survey Insights</div>',
                    unsafe_allow_html=True)
        co1,co2 = st.columns(2)
        with co1:
            ac = s_df["q1_age_range"].value_counts().reindex(AGE_ORDER).dropna()
            fig_a = go.Figure(go.Bar(x=ac.index,y=ac.values,
                marker=dict(color=ac.values,colorscale="Viridis"),
                text=ac.values,textposition="outside"))
            fig_a.update_xaxes(tickangle=-30)
            _L(fig_a,"Age Distribution",h=320,yaxis_title="Count")
            st.plotly_chart(fig_a, use_container_width=True)
        with co2: st.plotly_chart(chart_platform(s_df), use_container_width=True)
        co3,co4 = st.columns(2)
        with co3: st.plotly_chart(chart_concern_scatter(s_df),    use_container_width=True)
        with co4: st.plotly_chart(chart_behaviour_score(s_df),    use_container_width=True)
        if "q12_concern_data_use" in s_df.columns:
            lk = s_df["q12_concern_data_use"].value_counts().sort_index()
            fig_lk = go.Figure(go.Bar(
                x=[str(int(v)) for v in lk.index], y=lk.values,
                marker_color=[TEAL,TEAL,YELL,RED,RED][:len(lk)],
                text=lk.values, textposition="outside"))
            _L(fig_lk,"Q12: Concern About Data Use  (1=low → 5=high)",h=300,
               xaxis_title="Rating",yaxis_title="Count")
            st.plotly_chart(fig_lk, use_container_width=True)

    # TAB 5 – Explorer
    with tabs[5]:
        st.markdown('<div class="section-header">📋 Data Explorer</div>',
                    unsafe_allow_html=True)
        ds = st.selectbox("Dataset:", ["Survey Responses","Genre PII Profile",
                           "Risk Matrix","Affinity Matrix","Comments"])
        search = st.text_input("🔎 Filter rows:", "")
        if ds=="Survey Responses":
            cols = ["q1_age_range","q2_education_level","q3_digital_literacy",
                    "concern_index","behaviour_gap","behaviour_score",
                    "risk_label","has_youtube","uses_personalization"]
            df_show = s_df[[c for c in cols if c in s_df.columns]]
        elif ds=="Genre PII Profile":
            df_show = g_pii.drop(columns=["risk_prior"],errors="ignore")
        elif ds=="Risk Matrix":
            df_show = r_mat
        elif ds=="Affinity Matrix":
            df_show = a_mat
        else:
            cols = ["comment_id","genre_name","text_clean","pii_count",
                    "max_sensitivity","sentiment_score","like_count"]
            df_show = c_df[[c for c in cols if c in c_df.columns]]

        if search:
            mask = df_show.astype(str).apply(
                lambda col: col.str.contains(search,case=False,na=False)).any(axis=1)
            df_show = df_show[mask]

        st.markdown(f"**{len(df_show):,} rows · {len(df_show.columns)} cols**")
        st.dataframe(df_show, use_container_width=True, height=450)
        st.download_button("⬇️ Download CSV", df_show.to_csv(index=False),
            file_name=f"{ds.replace(' ','_').lower()}.csv", mime="text/csv")

    st.markdown("""
    <hr style="border-color:#30363D;margin:2rem 0 0.5rem 0;">
    <div style="text-align:center;color:#484F58;font-size:0.7rem;">
        Privacy Perception Study · Results loaded from disk — no live DB calls
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

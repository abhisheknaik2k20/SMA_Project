# Privacy Perception Study — Streamlit Dashboard v2

Multimodal AI analytics dashboard for social media privacy research.

## Files

```
privacy_dashboard/
├── app.py              ← Streamlit dashboard (this is the main file)
├── pipeline_v2.py      ← Your pipeline code (place here)
├── requirements.txt
├── run.sh              ← Launch script
└── pipeline_cache/     ← Auto-created when data is loaded
    ├── results.json
    └── meta.json
```

## Quick Start

### Option A — Demo with mock data (no Supabase needed)
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
streamlit run app.py
```
Then click **"⚡ Load Mock Data"** in the UI.

### Option B — Run with real data
1. Place your `pipeline_v2.py` (the full pipeline code) in the same folder as `app.py`
2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```
3. Launch:
   ```bash
   bash run.sh
   # or
   streamlit run app.py
   ```
4. Enter your **Supabase URL + Key** in the UI and click **"▶ Run Full Pipeline"**

## How it works

```
App starts
    │
    ▼
pipeline_cache/results.json exists?
    │                    │
   YES                   NO
    │                    │
    ▼                    ▼
Load dashboard       Show runner UI
(instant)                │
                         ├─ Mock Data button → generate + cache → reload
                         │
                         └─ Run Pipeline → call MultimodalPrivacyPipelineV2
                                         → save results to JSON cache
                                         → reload → show dashboard
```

## Dashboard Pages

| Page | Content |
|------|---------|
| 📊 Overview | KPIs, genre risk chart, top-10 risk pairs, model metrics, modality weights |
| 🎯 Genre × Age Risk | Susceptibility heatmap + affinity heatmap |
| 🔍 PII Analysis | PII type breakdown (Financial / Contact / Identity / Relational), data table |
| ⚗️ Causal Inference | PSM ATT forest plot, significant effects table |
| 📈 Longitudinal | Concern slope distribution, trajectory classification, user table |
| 🤖 Model | Feature importances, GBM metrics, fusion attention weights |

## Cache Management

- Cache lives in `pipeline_cache/results.json`
- To force a re-run: click **"🗑️ Clear Cache & Re-run"** in the sidebar, or just delete the file
- Cache stores all serialisable outputs (DataFrames as records, numpy arrays as lists)

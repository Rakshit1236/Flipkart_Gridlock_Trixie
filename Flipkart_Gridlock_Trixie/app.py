import streamlit as st
import pandas as pd
import pickle, os, sys, time, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_CSV, HEATMAP_DIR, REPORT_DIR, PRED_DIR, OUTPUT_DIR, MODEL_DIR

CACHE_FILE = os.path.join(OUTPUT_DIR, "pipeline_cache.pkl")

st.set_page_config(
    page_title="Bengaluru Parking Intelligence",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)


def run_full_pipeline():
    from src.data_pipeline import run_pipeline
    from src.clustering import run_clustering
    from src.traffic_impact import run_impact
    from src.scoring import run_scoring
    from src.predictive_model import run_prediction
    from src.analytics import compute_xai_breakdown, build_adjacency, generate_recommendations, compute_early_warning

    t0 = time.time()
    df = run_pipeline()
    df, profiles, ts, clusterer = run_clustering(df)
    impact_df, hourly, ripples = run_impact(df, profiles)
    total_days = df["date"].nunique()
    priority_df, risk_df = run_scoring(profiles, impact_df, total_days)

    # Predictions
    preds, feature_df, feature_cols, r2 = run_prediction(df, profiles, n_trials=10)

    # Dispatch report
    date_str = str(pd.Timestamp.now().date())
    report_lines = [f"DISPATCH REPORT — {date_str}", "=" * 50, ""]
    for idx, row in priority_df.head(10).iterrows():
        report_lines.append(f"#{idx+1} {row['junction_name']} (Score: {row['priority_score']:.0f})")
        report_lines.append(f"   Area: {row.get('area', 'N/A')} | Road: {row.get('road_type', 'N/A')}")
        report_lines.append(f"   Speed Drop: {row.get('worst_speed_drop_pct', 0):.1f}% | Severity: {row.get('worst_severity', 'N/A')}")
        report_lines.append(f"   Peak Hours: {row.get('peak_hours', [])} | Violations: {row.get('total_violations', 0)}")
        report_lines.append("")
    with open(os.path.join(REPORT_DIR, f"dispatch_{date_str}.txt"), "w") as f:
        f.write("\n".join(report_lines))
    priority_df.to_csv(os.path.join(REPORT_DIR, f"priority_scores_{date_str}.csv"), index=False)

    # Advanced analytics
    xai = compute_xai_breakdown(df, impact_df, profiles)
    adj = build_adjacency(profiles)
    recs = generate_recommendations(priority_df, impact_df, risk_df)
    warnings = compute_early_warning(df, profiles)

    # Visualizations
    from src.visualization import generate_all_visualizations
    generate_all_visualizations(df, profiles, impact_df, priority_df)

    data = {
        "df": df, "profiles": profiles, "impact_df": impact_df,
        "priority_df": priority_df, "risk_df": risk_df,
        "xai": xai, "adjacency": adj, "recommendations": recs,
        "early_warnings": warnings, "predictions": preds,
    }

    # Cache to disk
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Pipeline complete in {time.time()-t0:.1f}s — cached to disk")
    return data


def load_cached_or_run():
    # Try loading from disk cache first (instant)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            return data, True
        except Exception:
            pass
    return None, False


# ── Sidebar ──
with st.sidebar:
    st.title("🅿️ Bengaluru Parking Intelligence")
    st.markdown("---")
    run_btn = st.button("Run Full Pipeline", type="primary", use_container_width=True)
    clear_cache = st.button("Clear Cache & Rerun", use_container_width=True)
    st.markdown("---")
    st.markdown("### Quick Stats")

if clear_cache:
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    st.cache_data.clear()
    st.rerun()

if run_btn:
    with st.spinner("Running full pipeline (~2 min)..."):
        data = run_full_pipeline()
    st.success("Pipeline complete!")
else:
    data, from_cache = load_cached_or_run()
    if data is None:
        st.warning("Click **Run Full Pipeline** to initialize the system.")
        st.stop()
    if from_cache:
        pass  # silently loaded from cache

# ── Update sidebar stats ──
with st.sidebar:
    st.metric("Hotspots", len(data["profiles"]))
    sev_col = "worst_severity"
    if sev_col in data["impact_df"].columns:
        st.metric("Critical", len(data["impact_df"][data["impact_df"][sev_col] == "CRITICAL"]))
    st.metric("Records", f"{len(data['df']):,}")

# ── Tabs ──
st.title("Bengaluru Parking Intelligence Platform")
st.markdown("*Forecast → Explain → Simulate → Recommend*")

tab_main, tab_scenario, tab_propagation, tab_insights, tab_actions = st.tabs([
    "Overview & Maps", "What-If Simulator", "Congestion Propagation",
    "Insights (XAI + PRI)", "Actions (Warnings + Dispatch)"
])

predictions_df = data.get("predictions", None)
if predictions_df is None:
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "predictions_*.csv")))
    if pred_files:
        predictions_df = pd.read_csv(pred_files[-1])

with tab_main:
    from dashboard.tab_main import render_overview, render_heatmaps, render_predictions, render_dispatch
    render_overview(data["priority_df"], data["impact_df"])
    st.markdown("---")
    tab_map, tab_pred, tab_disp = st.tabs(["Heatmaps", "Tomorrow's Forecast", "Dispatch Report"])
    with tab_map:
        render_heatmaps()
    with tab_pred:
        render_predictions(predictions_df)
    with tab_disp:
        render_dispatch()

with tab_scenario:
    from dashboard.tab_scenario import render as render_scenario
    render_scenario(data["profiles"], data["impact_df"], data["df"])

with tab_propagation:
    from dashboard.tab_propagation import render as render_propagation
    render_propagation(data["profiles"], data["adjacency"], data["df"])

with tab_insights:
    from dashboard.tab_insights import render_insights
    render_insights(data["xai"], data["risk_df"])

with tab_actions:
    from dashboard.tab_actions import render_actions
    render_actions(data["early_warnings"], data["recommendations"], data["df"], data["profiles"])

st.markdown("---")
st.markdown("*Bengaluru Parking Intelligence Platform — Build by Trixie*")

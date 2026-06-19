import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import REPORT_DIR, HEATMAP_DIR, PRED_DIR, OUTPUT_DIR
from src.visualization import create_predictions_heatmap

st.set_page_config(page_title="AI Parking Intelligence", layout="wide", page_icon="🅿")
st.title("AI-Driven Illegal Parking Intelligence System")
st.subheader("Bengaluru Traffic Department - Real-Time Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Hotspot Map", "Priority Rankings", "Impact Analysis", "Dispatch Report"
])

with tab1:
    st.header("System Overview")
    col1, col2, col3, col4 = st.columns(4)

    priority_files = sorted(glob.glob(os.path.join(REPORT_DIR, "priority_scores_*.csv")))
    if priority_files:
        latest = pd.read_csv(priority_files[-1])
        col1.metric("Total Hotspots", len(latest))
        col2.metric("Critical Hotspots", len(latest[latest["impact_severity"] == "CRITICAL"]))
        col3.metric("Avg Speed Drop", "{:.0f}%".format(latest["worst_speed_drop_pct"].mean()))
        col4.metric("Total Vehicle-Hours Lost", "{:.0f}".format(latest["total_vehicle_hours_lost"].sum()))

        st.subheader("Violation Distribution by Road Type")
        if "road_type" in latest.columns:
            road_counts = latest["road_type"].value_counts()
            st.bar_chart(road_counts)

        st.subheader("Impact Severity Distribution")
        severity_counts = latest["impact_severity"].value_counts()
        st.bar_chart(severity_counts)
    else:
        st.info("Run main.py first to generate data.")

with tab2:
    st.header("Hotspot Heatmap")
    heatmap_files = sorted(glob.glob(os.path.join(HEATMAP_DIR, "*.html")))
    if heatmap_files:
        selected = st.selectbox("Select heatmap:", [os.path.basename(f) for f in heatmap_files])
        map_path = os.path.join(HEATMAP_DIR, selected)
        with open(map_path, "r") as f:
            st.iframe(f.read(), height=600)
    else:
        st.info("No heatmaps found. Run main.py first.")

with tab3:
    st.header("Priority Rankings")
    if priority_files:
        latest = pd.read_csv(priority_files[-1])
        display_cols = ["label", "priority_score", "total_violations", "worst_speed_drop_pct",
                        "impact_severity", "total_vehicle_hours_lost", "unique_days"]
        available = [c for c in display_cols if c in latest.columns]
        st.dataframe(
            latest[available].style.background_gradient(subset=["priority_score"], cmap="RdYlGn_r"),
            width="stretch",
        )

        st.subheader("Priority Score Components")
        component_cols = ["frequency_score", "impact_score", "urgency_score", "criticality_score"]
        avail_comp = [c for c in component_cols if c in latest.columns]
        if avail_comp:
            st.bar_chart(latest[avail_comp].head(10))

with tab4:
    st.header("Impact Analysis")
    if priority_files:
        latest = pd.read_csv(priority_files[-1])
        if "worst_speed_drop_pct" in latest.columns:
            st.subheader("Speed Reduction by Hotspot")
            chart_data = latest[["label", "worst_speed_drop_pct"]].head(15)
            chart_data = chart_data.set_index("label")
            st.bar_chart(chart_data)

            st.subheader("Vehicle-Hours Lost by Hotspot")
            vhl_data = latest[["label", "total_vehicle_hours_lost"]].head(15)
            vhl_data = vhl_data.set_index("label")
            st.bar_chart(vhl_data)

with tab5:
    st.header("Daily Dispatch Report")
    dispatch_files = sorted(glob.glob(os.path.join(REPORT_DIR, "dispatch_*.txt")))
    if dispatch_files:
        selected_dispatch = st.selectbox("Select report:", [os.path.basename(f) for f in dispatch_files])
        with open(os.path.join(REPORT_DIR, selected_dispatch), "r") as f:
            st.code(f.read(), language=None)
    else:
        st.info("No dispatch reports found. Run main.py first.")

    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "predictions_*.csv")))
    if pred_files:
        st.subheader("Predictions Heatmap — Tomorrow's High-Risk Locations")
        preds = pd.read_csv(pred_files[-1])
        
        # Generate and display predictions heatmap
        heatmap_path = create_predictions_heatmap(preds)
        with open(heatmap_path, "r") as f:
            st.iframe(f.read(), height=600)
        
        st.subheader("Tomorrow's Predictions — Where Traffic Will Be Affected")
        top_active = preds[preds["activation_probability"] >= 0.5].copy()
        if len(top_active) > 0:
            st.warning("{} hotspots predicted to be ACTIVE tomorrow (>= 50% probability)".format(len(top_active)))

            for _, row in top_active.head(15).iterrows():
                label = row.get("label", "Cluster " + str(int(row["cluster_id"])))
                prob = row["activation_probability"]
                est = int(row.get("estimated_violations_tomorrow", 0))
                peak = row.get("peak_hours", "N/A")
                vtype = row.get("dominant_violation", "N/A")
                speed_drop = row.get("worst_speed_drop_pct", 0)
                sev = row.get("impact_severity", "N/A")

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Location", label)
                col_b.metric("Activation Probability", "{:.0f}%".format(prob * 100))
                col_c.metric("Est. Violations", str(est))
                col_d.metric("Speed Impact", "-{:.0f}%".format(speed_drop) if speed_drop > 0 else "N/A")

                detail_col1, detail_col2, detail_col3 = st.columns(3)
                detail_col1.caption("Peak Hours: {}".format(peak))
                detail_col2.caption("Violation Type: {}".format(vtype))
                detail_col3.caption("Severity: {}".format(sev))
                st.divider()
        else:
            st.info("No hotspots predicted to be highly active tomorrow.")

        show_all = st.checkbox("Show all predicted hotspots")
        if show_all:
            display_pred = [c for c in ["label", "activation_probability", "predicted_violation_count",
                                         "estimated_violations_tomorrow",
                                         "violation_count", "dominant_violation", "peak_hours",
                                         "impact_severity", "worst_speed_drop_pct"]
                            if c in preds.columns]
            st.dataframe(preds[display_pred], width="stretch")

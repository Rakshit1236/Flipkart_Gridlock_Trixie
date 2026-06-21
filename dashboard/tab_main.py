import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
import branca.colormap
import os, glob

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HEATMAP_DIR, REPORT_DIR, PRED_DIR


def _latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def render_overview(priority_df, impact_df):
    st.subheader("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Hotspots", len(priority_df))
    col2.metric("Critical Hotspots", len(priority_df[priority_df.get("worst_severity", "LOW") == "CRITICAL"]) if "worst_severity" in priority_df.columns else "N/A")
    col3.metric("Avg Speed Drop", f"{impact_df['worst_speed_drop_pct'].mean():.1f}%" if "worst_speed_drop_pct" in impact_df.columns else "N/A")
    col4.metric("Total Vehicle-Hrs Lost", f"{impact_df.get('total_vehicle_hours_lost', pd.Series([0])).sum():,.0f}" if "total_vehicle_hours_lost" in impact_df.columns else "N/A")

    if "road_type" in priority_df.columns:
        fig1 = px.bar(priority_df.groupby("road_type").size().reset_index(name="count"),
                      x="road_type", y="count", color="road_type",
                      title="Violations by Road Type")
        st.plotly_chart(fig1, use_container_width=True)

    if "worst_severity" in impact_df.columns:
        fig2 = px.pie(impact_df, names="worst_severity", title="Severity Distribution",
                      color="worst_severity", hole=0.3)
        st.plotly_chart(fig2, use_container_width=True)


def render_heatmaps():
    st.subheader("Interactive Heatmaps")
    maps = sorted(glob.glob(os.path.join(HEATMAP_DIR, "*.html")))
    if not maps:
        st.info("No heatmap files found. Run the pipeline first.")
        return
    labels = [os.path.basename(f) for f in maps]
    choice = st.selectbox("Select a map", labels)
    if choice:
        idx = labels.index(choice)
        with open(maps[idx], "r") as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)


def render_predictions(predictions_df):
    st.subheader("Tomorrow's Traffic Forecast")
    if predictions_df is None or len(predictions_df) == 0:
        st.info("No predictions available. Run the pipeline first.")
        return

    high_conf = predictions_df[predictions_df["activation_probability"] >= 50]
    st.metric("Predicted Active Hotspots", len(high_conf))

    fig = px.bar(predictions_df.head(20), x="junction_name", y="predicted_violations",
                 color="activation_probability", color_continuous_scale="RdYlGn_r",
                 title="Top 20 Predicted Hotspots — Tomorrow",
                 labels={"predicted_violations": "Predicted Violations", "junction_name": "Location"})
    st.plotly_chart(fig, use_container_width=True)

    for _, row in high_conf.head(10).iterrows():
        with st.expander(f"{row['junction_name']} — {row['predicted_violations']} violations ({row['confidence_pct']}% CI)"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Probability", f"{row['activation_probability']:.0f}%")
            c2.metric("Est. Violations", f"{row['predicted_violations']}")
            c3.metric("Confidence", f"{row['confidence_pct']}%")
            c4.metric("Severity", row.get("dominant_violation", "N/A"))
            st.caption(f"Peak Hours: {row.get('peak_hours', 'N/A')} | Road: {row.get('road_type', 'N/A')}")

    if st.checkbox("Show all predictions as table"):
        st.dataframe(predictions_df[["cluster_id", "junction_name", "predicted_violations",
                                      "activation_probability", "confidence_pct", "ci_lower", "ci_upper"]])


def render_dispatch():
    st.subheader("Dispatch Report")
    reports = sorted(glob.glob(os.path.join(REPORT_DIR, "dispatch_*.txt")))
    if not reports:
        st.info("No dispatch reports found. Run the pipeline first.")
        return
    labels = [os.path.basename(f) for f in reports]
    choice = st.selectbox("Select report", labels)
    if choice:
        idx = labels.index(choice)
        with open(reports[idx], "r") as f:
            st.code(f.read(), language=None)

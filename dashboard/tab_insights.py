import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_xai(xai_df):
    st.markdown("### Explainable AI — Root Cause Attribution")
    st.markdown("For each hotspot, see **why** congestion occurs — broken down into percentage contributors.")

    if xai_df is None or len(xai_df) == 0:
        st.info("No XAI data available.")
        return

    location = st.selectbox("Select hotspot", xai_df["junction_name"].tolist(), key="xai_select")
    row = xai_df[xai_df["junction_name"] == location]
    if len(row) == 0:
        return
    row = row.iloc[0]
    breakdown = row["breakdown"]

    st.markdown(f"**Dominant Factor:** `{row['dominant_factor'].replace('_', ' ').title()}`")

    labels = [k.replace("_", " ").title() for k in breakdown.keys()]
    values = list(breakdown.values())
    colors = ["#ef4444", "#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4,
                                     marker_colors=colors, textinfo="label+percent")])
        fig.update_layout(title="Root Cause Breakdown", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure(data=[go.Bar(x=values, y=labels, orientation="h",
                                      marker_color=colors[:len(labels)])])
        fig2.update_layout(title="Contributor Weights (%)", height=350, xaxis_title="%")
        st.plotly_chart(fig2, use_container_width=True)


def render_risk_index(risk_df):
    st.markdown("### Parking Risk Index (PRI)")
    st.markdown("**Proprietary Metric:** `0.4 × Illegal Parking + 0.3 × Density + 0.2 × Road Importance + 0.1 × Event Score`")

    if risk_df is None or len(risk_df) == 0:
        st.info("No risk index data available.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg PRI", f"{risk_df['parking_risk_index'].mean():.1f}")
    c2.metric("Max PRI", f"{risk_df['parking_risk_index'].max():.1f}")
    c3.metric("High Risk (PRI ≥ 70)", len(risk_df[risk_df["parking_risk_index"] >= 70]))

    fig = px.histogram(risk_df, x="parking_risk_index", nbins=30,
                       color_discrete_sequence=["#ef4444"],
                       title="PRI Distribution Across Hotspots")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 15 Highest Risk Locations")
    top = risk_df.nlargest(15, "parking_risk_index")[
        ["junction_name", "area", "parking_risk_index", "total_violations", "is_chronic", "road_type"]
    ].copy()
    top.columns = ["Location", "Area", "PRI Score", "Total Violations", "Chronic", "Road Type"]
    st.dataframe(top.style.background_gradient(subset=["PRI Score"], cmap="RdYlGn_r"),
                 use_container_width=True)


def render_insights(xai_df, risk_df):
    tab1, tab2 = st.tabs(["Explainable AI", "Parking Risk Index"])
    with tab1:
        render_xai(xai_df)
    with tab2:
        render_risk_index(risk_df)

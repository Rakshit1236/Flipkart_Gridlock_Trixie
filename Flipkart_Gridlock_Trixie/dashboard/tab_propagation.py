import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import build_adjacency, simulate_propagation


def render(profiles, adjacency, df):
    st.subheader("Congestion Propagation Forecast")
    st.markdown("See how an incident at one hotspot **cascades** to neighboring areas over time.")

    col1, col2 = st.columns([1, 2])
    with col1:
        source = st.selectbox("Incident origin", profiles["junction_name"].tolist())
        minutes_ahead = st.slider("Forecast horizon (minutes)", 15, 120, 60, step=15)
        start_hour = st.slider("Start hour", 6, 22, 8)

    prof_row = profiles[profiles["junction_name"] == source]
    if len(prof_row) == 0:
        st.warning("Select a valid hotspot")
        return

    source_id = prof_row.iloc[0]["cluster_id"]
    chain = simulate_propagation(source_id, adjacency, profiles,
                                  start_hour=start_hour, minutes_ahead=minutes_ahead)

    if not chain:
        st.info("No propagation detected within this horizon.")
        return

    # Timeline visualization
    st.markdown("### Cascade Timeline")
    timeline_df = pd.DataFrame(chain)

    fig = px.scatter(timeline_df, x="minute_offset", y="speed_drop_pct",
                     size=[max(5, s) for s in timeline_df["speed_drop_pct"]],
                     color="speed_drop_pct", color_continuous_scale="RdYlGn_r",
                     hover_name="junction_name",
                     title=f"Speed Drop Propagation from {source}",
                     labels={"minute_offset": "Minutes After Incident",
                             "speed_drop_pct": "Speed Drop %"})
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Incident Start")
    st.plotly_chart(fig, use_container_width=True)

    # Map view
    st.markdown("### Propagation Map")
    m = folium.Map(location=[prof_row.iloc[0]["centroid_lat"], prof_row.iloc[0]["centroid_lon"]],
                   zoom_start=13, tiles="CartoDB dark_matter")

    # Source marker
    folium.Marker(
        [prof_row.iloc[0]["centroid_lat"], prof_row.iloc[0]["centroid_lon"]],
        popup=f"<b>SOURCE:</b> {source}<br>Start: {start_hour}:00",
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
    ).add_to(m)

    # Affected nodes
    for step in chain:
        if step["cluster_id"] == source_id:
            continue
        prof_match = profiles[profiles["cluster_id"] == step["cluster_id"]]
        if len(prof_match) == 0:
            continue
        p = prof_match.iloc[0]
        drop = step["speed_drop_pct"]
        color = "red" if drop >= 40 else ("orange" if drop >= 25 else ("yellow" if drop >= 10 else "green"))
        folium.CircleMarker(
            [p["centroid_lat"], p["centroid_lon"]],
            radius=max(5, drop / 4),
            color=color, fill=True, fill_opacity=0.7,
            popup=f"<b>{step['junction_name']}</b><br>T+{step['minute_offset']}min<br>Speed Drop: {drop}%<br>Time: {step['timestamp']}",
        ).add_to(m)

    st_folium(m, width=700, height=500)

    # Chain table
    st.markdown("### Affected Hotspots")
    display_df = timeline_df[["timestamp", "junction_name", "speed_drop_pct", "minute_offset"]].copy()
    display_df.columns = ["Time", "Location", "Speed Drop %", "Minutes After"]
    st.dataframe(display_df, use_container_width=True)

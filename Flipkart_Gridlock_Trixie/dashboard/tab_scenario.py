import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scenario_simulator import run_scenario, WEATHER_MULTIPLIERS, DAY_TYPE_MULTIPLIERS


def render(profiles, impact_df, df):
    st.subheader("What-If Scenario Simulator")
    st.markdown("Adjust variables below and see **instant impact** on traffic metrics across Bengaluru hotspots.")

    col1, col2, col3 = st.columns(3)
    with col1:
        vehicle_reduction = st.slider(
            "Remove illegally parked vehicles", 0, 200, 0,
            help="Simulate removing parked vehicles from the road"
        )
    with col2:
        weather = st.selectbox("Weather condition", list(WEATHER_MULTIPLIERS.keys()), index=0)
    with col3:
        day_type = st.selectbox("Day type", list(DAY_TYPE_MULTIPLIERS.keys()), index=0)

    focus_options = ["All Hotspots"] + profiles["junction_name"].tolist()
    focus = st.selectbox("Focus on specific hotspot", focus_options)
    focus_id = None
    if focus != "All Hotspots":
        row = profiles[profiles["junction_name"] == focus]
        if len(row) > 0:
            focus_id = row.iloc[0]["cluster_id"]

    if st.button("Run Scenario", type="primary"):
        with st.spinner("Simulating..."):
            result = run_scenario(profiles, impact_df, df,
                                  vehicle_reduction=vehicle_reduction,
                                  weather=weather, day_type=day_type,
                                  focus_cluster_id=focus_id)

        if len(result) == 0:
            st.warning("No results for this scenario.")
            return

        st.success(f"Scenario complete — {len(result)} hotspots affected")

        # Summary metrics
        st.markdown("### City-Wide Impact")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Speed Change", f"{result['speed_change'].mean():+.1f} km/h")
        c2.metric("Avg Delay Reduction", f"{result['delay_change_min'].mean():+.1f} min")
        c3.metric("Vehicle-Hrs Saved", f"{result['vhl_change'].sum():+,.0f}")
        c4.metric("Violations Reduced", f"{result['violations_change'].sum():+}")

        # Before/after comparison
        st.markdown("### Before vs After")
        compare = result[["junction_name", "baseline_speed_kmh", "scenario_speed_kmh",
                          "baseline_delay_min", "scenario_delay_min",
                          "baseline_queue_length", "scenario_queue_length"]].head(15)
        compare.columns = ["Location", "Base Speed", "New Speed", "Base Delay", "New Delay",
                           "Base Queue", "New Queue"]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline", x=compare["Location"], y=compare["Base Speed"],
                             marker_color="#ef4444"))
        fig.add_trace(go.Bar(name="Scenario", x=compare["Location"], y=compare["New Speed"],
                             marker_color="#22c55e"))
        fig.update_layout(barmode="group", title="Speed Comparison (km/h)")
        st.plotly_chart(fig, use_container_width=True)

        # Hotspot detail table
        st.markdown("### Detailed Results")
        detail = result[["junction_name", "area", "baseline_speed_drop_pct", "scenario_speed_drop_pct",
                         "baseline_violations", "scenario_violations",
                         "baseline_delay_min", "scenario_delay_min",
                         "delay_change_min", "violations_change"]].copy()
        detail.columns = ["Location", "Area", "Base Drop%", "New Drop%", "Base Viol.",
                          "New Viol.", "Base Delay", "New Delay", "Delay Saved", "Viol. Reduced"]
        st.dataframe(detail, use_container_width=True)

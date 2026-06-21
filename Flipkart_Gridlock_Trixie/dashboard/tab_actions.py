import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time as _time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import compute_early_warning


def render_early_warning(df, profiles):
    st.markdown("### Live Traffic Early Warning System")
    st.markdown("Predictions from **your selected time** — warns when congestion is likely.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Prediction base time:**")
        use_now = st.checkbox("Use current time", value=True)
        if use_now:
            base_time = datetime.now()
            st.success(f"Now: **{base_time.strftime('%I:%M %p')}**")
        else:
            sel_hour = st.slider("Hour", 0, 23, datetime.now().hour)
            sel_min = st.selectbox("Minute", [0, 5, 10, 15, 30, 45], index=0)
            base_time = datetime.now().replace(hour=sel_hour, minute=sel_min, second=0)
            st.info(f"Base time: **{base_time.strftime('%I:%M %p')}**")

    horizons = [15, 30, 60]
    with col2:
        st.markdown("**Forecast windows:**")
        selected = st.multiselect("Select horizons", horizons, default=horizons, format_func=lambda x: f"+{x} min")
        if not selected:
            selected = horizons

    with st.spinner("Computing predictions..."):
        warnings_df = compute_early_warning(df, profiles, base_time=base_time, prediction_horizons=selected)

    if warnings_df is None or len(warnings_df) == 0:
        st.warning("No warnings generated.")
        return

    # Summary cards
    st.markdown("---")
    st.markdown("#### Threat Summary")
    summary_cols = st.columns(len(selected))
    for i, h in enumerate(sorted(selected)):
        subset = warnings_df[warnings_df["horizon_minutes"] == h]
        future_time = base_time + timedelta(minutes=h)
        high_count = len(subset[subset["threat_level"] == "HIGH"])
        med_count = len(subset[subset["threat_level"] == "MEDIUM"])
        total_threat = high_count + med_count
        with summary_cols[i]:
            label = f"by {future_time.strftime('%I:%M %p')} (+{h}m)"
            delta_color = "off" if total_threat == 0 else "inverse"
            st.metric(label, f"{high_count} HIGH", delta=f"{med_count} MEDIUM", delta_color=delta_color)

    # Per-horizon details
    for h in sorted(selected):
        future_time = base_time + timedelta(minutes=h)
        subset = warnings_df[warnings_df["horizon_minutes"] == h].copy()
        high_threat = subset[subset["threat_level"].isin(["HIGH", "MEDIUM"])]

        st.markdown(f"---")
        st.markdown(f"#### +{h} min — by {future_time.strftime('%I:%M %p')}")
        st.caption(f"Rush hour at forecast: {'Yes' if subset['rush_hour_at_forecast'].any() else 'No'}")

        if len(high_threat) == 0:
            st.success("No significant threats predicted.")
            continue

        # Map
        fig = px.scatter(high_threat, x="centroid_lat", y="centroid_lon",
                         size="congestion_probability",
                         color="threat_level",
                         color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b"},
                         hover_name="junction_name",
                         size_max=20,
                         title=f"Congestion Risk at {future_time.strftime('%I:%M %p')}")
        st.plotly_chart(fig, use_container_width=True)

        # Alert table
        alert = high_threat[["junction_name", "congestion_probability", "threat_level", "predicts_for", "rush_hour_at_forecast"]].copy()
        alert.columns = ["Location", "Probability %", "Threat", "At Time", "Rush Hour?"]
        alert = alert.sort_values("Probability %", ascending=False)

        def highlight_threat(val):
            if val == "HIGH":
                return "background-color: #7f1d1d; color: white"
            elif val == "MEDIUM":
                return "background-color: #78350f; color: white"
            return ""
        st.dataframe(alert.style.map(highlight_threat, subset=["Threat"]), use_container_width=True)


def render_recommendations(recs_df):
    st.markdown("### Action Recommendation Engine")
    st.markdown("Transforms hotspot data into **actionable dispatch decisions** with expected impact.")

    if recs_df is None or len(recs_df) == 0:
        st.info("No recommendations available.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Recommendations", len(recs_df))
    c2.metric("Avg Officers Needed", f"{recs_df['deploy_officers'].mean():.1f}")
    c3.metric("Avg Delay Reduction", f"{recs_df['expected_delay_reduction_pct'].mean():.1f}%")

    for _, row in recs_df.head(10).iterrows():
        with st.expander(f"**{row['junction_name']}** — {row['urgency']} | Score: {row['priority_score']:.0f}"):
            st.markdown(f"**:red[{row['recommended_action']}]**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Deploy Officers", f"{row['deploy_officers']}")
            c2.metric("Dispatch Time", row["dispatch_time"])
            c3.metric("Expected Delay Reduction", f"{row['expected_delay_reduction_pct']:.0f}%")
            c4.metric("Current Speed Drop", f"{row['current_speed_drop_pct']:.0f}%")
            st.caption(f"Parking Risk Index: {row.get('parking_risk_index', 0):.0f} | Vehicle-Hrs Lost: {row.get('vehicle_hours_lost', 0):.0f}")


def render_actions(warnings_df, recs_df, df, profiles):
    tab1, tab2 = st.tabs(["Live Early Warning", "Action Recommendations"])
    with tab1:
        render_early_warning(df, profiles)
    with tab2:
        render_recommendations(recs_df)

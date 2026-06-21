import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import branca.colormap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HEATMAP_DIR, REPORT_DIR


def create_hotspot_heatmap(profiles):
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB positron")
    colormap = branca.colormap.LinearColormap(["green", "yellow", "orange", "red"], vmin=0, vmax=profiles["total_violations"].max())
    for _, p in profiles.iterrows():
        color = colormap(min(p["total_violations"], profiles["total_violations"].max()))
        popup_html = f"<b>{p['junction_name']}</b><br>Violations: {p['total_violations']}<br>Peak: {p['peak_hours']}<br>Type: {p['dominant_violation']}"
        folium.CircleMarker(
            [p["centroid_lat"], p["centroid_lon"]],
            radius=max(3, min(15, p["total_violations"] / 50)),
            color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)
    m.save(os.path.join(HEATMAP_DIR, "hotspot_heatmap.html"))
    print("  hotspot_heatmap.html")


def create_impact_map(impact_df):
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")
    sev_colors = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}
    for _, r in impact_df.iterrows():
        color = sev_colors.get(r.get("worst_severity", "LOW"), "gray")
        popup_html = f"<b>{r['junction_name']}</b><br>Speed Drop: {r['worst_speed_drop_pct']:.1f}%<br>Severity: {r.get('worst_severity', 'N/A')}<br>VHL: {r.get('total_vehicle_hours_lost', 0):.0f}"
        folium.CircleMarker(
            [r["centroid_lat"], r["centroid_lon"]],
            radius=max(3, min(15, r["worst_speed_drop_pct"] / 3)),
            color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)
    m.save(os.path.join(HEATMAP_DIR, "impact_map.html"))
    print("  impact_map.html")


def create_temporal_heatmap(df):
    pivot = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Violations"})
    ax.set_title("Violations by Day of Week and Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(os.path.join(HEATMAP_DIR, "temporal_heatmap.png"), dpi=150)
    plt.close()
    print("  temporal_heatmap.png")


def create_predictions_heatmap(preds):
    if preds is None or len(preds) == 0:
        return
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")
    colormap = branca.colormap.LinearColormap(["yellow", "orange", "red"], vmin=0, vmax=100)
    high = preds[preds["activation_probability"] >= 30]
    for _, r in high.iterrows():
        color = colormap(min(r["activation_probability"], 100))
        popup_html = f"<b>{r['junction_name']}</b><br>Predicted: {r['predicted_violations']}<br>Probability: {r['activation_probability']:.0f}%<br>Confidence: {r['confidence_pct']}%"
        folium.CircleMarker(
            [r["centroid_lat"], r["centroid_lon"]],
            radius=max(3, min(15, r["predicted_violations"] / 3)),
            color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)
    m.save(os.path.join(HEATMAP_DIR, "predictions_heatmap.html"))
    print("  predictions_heatmap.html")


def create_priority_bar_chart(priority_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    top10 = priority_df.head(10)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top10)))
    ax1.barh(top10["junction_name"], top10["priority_score"], color=colors)
    ax1.set_xlabel("Priority Score")
    ax1.set_title("Top 10 Priority Hotspots")
    ax1.invert_yaxis()
    if all(c in top10.columns for c in ["freq_score", "impact_score", "urgency_score", "criticality_score"]):
        x = np.arange(len(top10))
        w = 0.2
        ax2.bar(x - 1.5*w, top10["freq_score"], w, label="Frequency")
        ax2.bar(x - 0.5*w, top10["impact_score"], w, label="Impact")
        ax2.bar(x + 0.5*w, top10["urgency_score"], w, label="Urgency")
        ax2.bar(x + 1.5*w, top10["criticality_score"], w, label="Criticality")
        ax2.set_xticks(x)
        ax2.set_xticklabels(top10["junction_name"], rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Score")
        ax2.set_title("Score Components")
        ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "priority_chart.png"), dpi=150)
    plt.close()
    print("  priority_chart.png")


def generate_all_visualizations(df, profiles, impact_df, priority_df):
    print("[Viz] Generating visualizations...")
    t0 = time.time()
    create_hotspot_heatmap(profiles)
    create_impact_map(impact_df)
    create_temporal_heatmap(df)
    create_priority_bar_chart(priority_df)
    print(f"  Visualizations done in {time.time()-t0:.1f}s")

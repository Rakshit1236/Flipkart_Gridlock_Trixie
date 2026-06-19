import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import HEATMAP_DIR, REPORT_DIR


def create_hotspot_heatmap(cluster_profiles, output_path=None):
    if output_path is None:
        output_path = os.path.join(HEATMAP_DIR, "hotspot_heatmap.html")

    print("[viz] Creating hotspot heatmap...")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB positron")

    colormap = cm.LinearColormap(
        colors=["green", "yellow", "orange", "red"],
        vmin=cluster_profiles["total_violations"].min(),
        vmax=cluster_profiles["total_violations"].max(),
        caption="Violation Count"
    )
    colormap.add_to(m)

    for _, row in cluster_profiles.iterrows():
        color = colormap(row["total_violations"])
        popup_html = (
            "<div style='width:250px'>"
            "<h4>{}</h4>"
            "<b>Total Violations:</b> {}<br>"
            "<b>Peak Hours:</b> {}<br>"
            "<b>Peak Day:</b> {}<br>"
            "<b>Dominant Type:</b> {}<br>"
            "<b>Chronic:</b> {}<br>"
            "<b>Avg Duration:</b> {:.0f} min"
            "</div>"
        ).format(
            row["label"], row["total_violations"], row["peak_hours"],
            row["peak_day"], row["dominant_violation"],
            "Yes" if row["chronic"] else "No", row["avg_duration_minutes"]
        )
        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=max(5, min(row["total_violations"] / 50, 30)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    m.save(output_path)
    print("  Saved to {}".format(output_path))
    return output_path


def create_impact_map(impact_df, output_path=None):
    if output_path is None:
        output_path = os.path.join(HEATMAP_DIR, "impact_map.html")

    print("[viz] Creating impact map...")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")

    severity_colors = {
        "CRITICAL": "red",
        "HIGH": "orange",
        "MEDIUM": "yellow",
        "LOW": "green",
        "NONE": "gray",
    }

    for _, row in impact_df.iterrows():
        color = severity_colors.get(row["impact_severity"], "gray")
        popup_html = (
            "<div style='width:280px'>"
            "<h4 style='color:{}'>{}</h4>"
            "<b>Speed Drop:</b> {:.0f}%<br>"
            "<b>Estimated Speed:</b> {:.0f} km/h<br>"
            "<b>Vehicle-Hours Lost:</b> {:.1f}<br>"
            "<b>Impact Severity:</b> {}<br>"
            "<b>Total Lanes:</b> {}<br>"
            "<b>Worst Hour:</b> {}:00"
            "</div>"
        ).format(
            color, row["label"], row["worst_speed_drop_pct"],
            row["worst_estimated_speed_kmh"], row["total_vehicle_hours_lost"],
            row["impact_severity"], row["total_lanes"], row["worst_hour"]
        )
        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=max(5, row["worst_speed_drop_pct"] / 3),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    m.save(output_path)
    print("  Saved to {}".format(output_path))
    return output_path


def create_temporal_heatmap(df, output_path=None):
    if output_path is None:
        output_path = os.path.join(HEATMAP_DIR, "temporal_heatmap.html")

    print("[viz] Creating temporal heatmap...")
    pivot = df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
    pivot_table = pivot.pivot_table(index="day_of_week", columns="hour", values="count", fill_value=0)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot_table.index = [day_labels[i] for i in pivot_table.index]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot_table, cmap="YlOrRd", annot=False, fmt="d", linewidths=0.5, ax=ax)
    ax.set_title("Illegal Parking Violations by Day of Week and Hour", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    fig.savefig(output_path.replace(".html", ".png"), dpi=150, bbox_inches="tight")
    plt.close()

    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11)
    heat_data = []
    for _, row in df.iterrows():
        heat_data.append([row["latitude"], row["longitude"], row["severity_weight"]])

    HeatMap(heat_data[:5000], radius=10, blur=15, max_zoom=13).add_to(m)
    m.save(output_path)
    print("  Saved to {}".format(output_path))
    return output_path


def create_priority_bar_chart(priority_df, output_path=None):
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, "priority_chart.png")

    print("[viz] Creating priority bar chart...")
    top10 = priority_df.head(10).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.RdYlGn_r(top10["priority_score"] / 100)
    axes[0].barh(range(len(top10)), top10["priority_score"], color=colors)
    axes[0].set_yticks(range(len(top10)))
    axes[0].set_yticklabels(top10["label"], fontsize=8)
    axes[0].set_xlabel("Priority Score")
    axes[0].set_title("Top 10 Enforcement Priority Hotspots")
    axes[0].invert_yaxis()

    score_components = top10[["frequency_score", "impact_score", "urgency_score", "criticality_score"]].values
    labels = ["Frequency", "Impact", "Urgency", "Criticality"]
    x = np.arange(len(top10))
    width = 0.2
    for i, (comp, label) in enumerate(zip(score_components.T, labels)):
        axes[1].bar(x + i * width, comp, width, label=label, alpha=0.8)
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(["#" + str(i + 1) for i in range(len(top10))])
    axes[1].set_ylabel("Score")
    axes[1].set_title("Priority Score Breakdown")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved to {}".format(output_path))
    return output_path


def create_predictions_heatmap(preds, output_path=None):
    if output_path is None:
        output_path = os.path.join(HEATMAP_DIR, "predictions_heatmap.html")

    print("[viz] Creating predictions heatmap...")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB positron")

    colormap = cm.LinearColormap(
        colors=["yellow", "orange", "red"],
        vmin=0.5,
        vmax=1.0,
        caption="Activation Probability"
    )
    colormap.add_to(m)

    for _, row in preds.iterrows():
        prob = row.get("activation_probability", 0)
        if prob < 0.3:
            continue

        label = row.get("label", "Cluster " + str(int(row.get("cluster_id", 0))))
        lat = row.get("centroid_lat", 12.9716)
        lon = row.get("centroid_lon", 77.5946)
        est = int(row.get("estimated_violations_tomorrow", 0))
        speed_drop = row.get("worst_speed_drop_pct", 0)
        vtype = row.get("dominant_violation", "N/A")
        peak = row.get("peak_hours", "N/A")
        sev = row.get("impact_severity", "N/A")

        color = colormap(prob)
        popup_html = (
            "<div style='width:280px'>"
            "<h4>{}</h4>"
            "<b>Activation Probability:</b> {:.0f}%<br>"
            "<b>Est. Violations Tomorrow:</b> {}<br>"
            "<b>Speed Impact:</b> -{:.0f}%<br>"
            "<b>Violation Type:</b> {}<br>"
            "<b>Peak Hours:</b> {}<br>"
            "<b>Severity:</b> {}"
            "</div>"
        ).format(label, prob * 100, est, speed_drop, vtype, peak, sev)

        radius = max(8, prob * 25)
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    m.save(output_path)
    print("  Saved to {}".format(output_path))
    return output_path


def generate_all_visualizations(df, cluster_profiles, impact_df, priority_df):
    print("[viz] Generating all visualizations...")
    create_hotspot_heatmap(cluster_profiles)
    create_impact_map(impact_df)
    create_temporal_heatmap(df)
    create_priority_bar_chart(priority_df)
    print("[viz] All visualizations generated!")

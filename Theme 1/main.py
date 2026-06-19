import pandas as pd
import numpy as np
import os
import sys
import time
import psutil
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import REPORT_DIR, PRED_DIR
from src.data_loader import load_and_clean
from src.preprocessing import preprocess
from src.hotspot_clustering import run_clustering
from src.traffic_impact import run_impact_analysis
from src.priority_scorer import compute_priority_scores
from src.predictive_model import run_prediction
from src.visualization import generate_all_visualizations


class ResourceMonitor:
    def __init__(self):
        self.log = []
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _monitor(self):
        while self.running:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            self.log.append({
                "timestamp": time.time(),
                "cpu_percent": cpu,
                "ram_percent": mem.percent,
                "ram_used_gb": mem.used / (1024**3),
            })

    def summary(self):
        if not self.log:
            return {}
        cpus = [e["cpu_percent"] for e in self.log]
        rams = [e["ram_percent"] for e in self.log]
        return {
            "cpu_avg": np.mean(cpus),
            "cpu_max": np.max(cpus),
            "ram_avg_pct": np.mean(rams),
            "ram_max_pct": np.max(rams),
            "ram_avg_gb": np.mean([e["ram_used_gb"] for e in self.log]),
            "samples": len(self.log),
        }


def generate_dispatch_report(priority_df, predictions, output_dir=None):
    if output_dir is None:
        output_dir = REPORT_DIR

    from datetime import datetime, timedelta
    tomorrow = datetime.now() + timedelta(days=1)
    date_str = tomorrow.strftime("%Y-%m-%d")
    day_name = tomorrow.strftime("%A")

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("   DAILY ENFORCEMENT DISPATCH RECOMMENDATION")
    report_lines.append("   Date: " + date_str + " (" + day_name + ")")
    report_lines.append("=" * 60)
    report_lines.append("")

    for rank, (_, row) in enumerate(priority_df.head(10).iterrows(), 1):
        peak_hours = row.get("peak_hours", [])
        if isinstance(peak_hours, str):
            import ast
            peak_hours = ast.literal_eval(peak_hours)
        dispatch_time = str(peak_hours[0]) + ":00" if peak_hours else "5:00 PM"

        speed_drop = row.get("worst_speed_drop_pct", 0)
        severity = row.get("impact_severity", "N/A")
        vhl = row.get("total_vehicle_hours_lost", 0)

        report_lines.append("#" + str(rank) + "  PRIORITY SCORE: " + str(row["priority_score"]) + "/100")
        report_lines.append("    Location: " + str(row["label"]))
        report_lines.append("    Action:  Send enforcement team to " + str(row["label"]) + " at " + dispatch_time)
        report_lines.append("    Reason:  Illegal parking there is currently dropping average traffic")
        report_lines.append("             speed by " + str(int(speed_drop)) + "%. " + str(row.get("dominant_violation", "N/A")) + " violations")
        report_lines.append("             are blocking an estimated " + str(round(row.get("blocked_lanes", 0), 1)) + " lanes.")
        report_lines.append("    Impact:  Severity=" + str(severity) + ", Vehicle-hours lost today: " + str(round(vhl, 1)))
        report_lines.append("    Violations: " + str(row["total_violations"]) + " total, " + str(row["unique_days"]) + " active days")
        report_lines.append("")

    report_lines.append("=" * 60)

    if len(predictions) > 0:
        top_pred = predictions.head(5)
        report_lines.append("   TOP PREDICTED ACTIVE HOTSPOTS TOMORROW")
        report_lines.append("-" * 60)
        for _, row in top_pred.iterrows():
            prob = row.get("activation_probability", 0)
            report_lines.append("    Cluster " + str(row["cluster_id"]) + ": " + str(round(prob * 100)) + "% probability")
        report_lines.append("")

    total_vhl = priority_df["total_vehicle_hours_lost"].sum() if "total_vehicle_hours_lost" in priority_df.columns else 0
    avg_drop = priority_df["worst_speed_drop_pct"].mean() if "worst_speed_drop_pct" in priority_df.columns else 0
    worst = priority_df.iloc[0] if len(priority_df) > 0 else None

    report_lines.append("   ESTIMATED CITY-WIDE IMPACT TODAY")
    report_lines.append("-" * 60)
    report_lines.append("   - Total vehicle-hours lost: " + str(int(total_vhl)))
    report_lines.append("   - Average speed reduction across hotspots: " + str(int(avg_drop)) + "%")
    if worst is not None:
        ws = str(worst["label"]) + " (-" + str(int(worst.get("worst_speed_drop_pct", 0))) + "% speed)"
        report_lines.append("   - Worst hotspot: " + ws)
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "dispatch_" + date_str + ".txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    priority_df.to_csv(os.path.join(output_dir, "priority_scores_" + date_str + ".csv"), index=False)
    if len(predictions) > 0:
        predictions.to_csv(os.path.join(PRED_DIR, "predictions_" + date_str + ".csv"), index=False)

    return report_text, report_path


def main():
    print("=" * 60)
    print("   AI-DRIVEN ILLEGAL PARKING INTELLIGENCE SYSTEM")
    print("   Bengaluru Traffic Department - Hackathon Solution")
    print("=" * 60)
    print()

    monitor = ResourceMonitor()
    monitor.start()
    pipeline_start = time.time()

    print("[STEP 1/6] Loading and cleaning data...")
    df = load_and_clean()
    print("  Clean dataset: " + str(len(df)) + " rows, " + str(df.shape[1]) + " columns")
    print()

    print("[STEP 2/6] Preprocessing...")
    df = preprocess(df)
    print()

    print("[STEP 3/6] Hotspot clustering...")
    df_clustered, profiles, ts_matrix, cluster_model = run_clustering(df)
    print()

    print("[STEP 4/6] Traffic impact quantification...")
    impact_df, hourly_details = run_impact_analysis(df_clustered, profiles)
    print()

    print("[STEP 5/6] Priority scoring...")
    total_days = df["date"].nunique()
    priority_df = compute_priority_scores(profiles, impact_df, total_days)
    print()

    print("[STEP 6/6] Predictive model...")
    predictions, pred_model, pred_features, pred_auc = run_prediction(df_clustered, profiles)
    print()

    print("[VISUALIZATION] Generating maps and charts...")
    from src.visualization import create_predictions_heatmap
    generate_all_visualizations(df_clustered, profiles, impact_df, priority_df)
    if len(predictions) > 0:
        create_predictions_heatmap(predictions)
    print()

    print("[REPORT] Generating dispatch report...")
    report_text, report_path = generate_dispatch_report(priority_df, predictions)
    print(report_text)
    print()

    pipeline_end = time.time()
    monitor.stop()
    resource_summary = monitor.summary()

    print("=" * 60)
    print("   RESOURCE USAGE SUMMARY")
    print("=" * 60)
    print("  Total pipeline time: " + str(round(pipeline_end - pipeline_start, 1)) + "s")
    cpu_avg = str(round(resource_summary.get("cpu_avg", 0), 1))
    cpu_max = str(round(resource_summary.get("cpu_max", 0), 1))
    print("  CPU avg: " + cpu_avg + "%  |  CPU max: " + cpu_max + "%")
    ram_avg = str(round(resource_summary.get("ram_avg_gb", 0), 2))
    ram_pct = str(round(resource_summary.get("ram_avg_pct", 0), 1))
    print("  RAM avg: " + ram_avg + " GB (" + ram_pct + "%)")
    print("  RAM max: " + str(round(resource_summary.get("ram_max_pct", 0), 1)) + "%")
    print("  Dataset: " + str(len(df)) + " records processed")
    print("  Hotspots found: " + str(len(profiles)))
    print("  Model AUC: " + str(round(pred_auc, 3)))
    print("  Report saved: " + report_path)
    print("=" * 60)


if __name__ == "__main__":
    main()

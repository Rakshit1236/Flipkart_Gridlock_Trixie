import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import REPORT_DIR


def run_cli(top_n=10, date=None):
    if date is None:
        date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    report_files = [f for f in os.listdir(REPORT_DIR) if f.startswith("dispatch_")]
    if not report_files:
        print("[cli] No dispatch reports found. Run main.py first.")
        return

    latest_report = sorted(report_files)[-1]
    report_path = os.path.join(REPORT_DIR, latest_report)

    with open(report_path, "r") as f:
        report_text = f.read()

    print(report_text)

    priority_files = [f for f in os.listdir(REPORT_DIR) if f.startswith("priority_scores_")]
    if priority_files:
        latest_priority = sorted(priority_files)[-1]
        priority_df = pd.read_csv(os.path.join(REPORT_DIR, latest_priority))
        print("\nTop {} dispatch recommendations for {}:".format(min(top_n, len(priority_df)), date))
        print("-" * 60)
        for rank, (_, row) in enumerate(priority_df.head(top_n).iterrows(), 1):
            peak_hours = row.get("peak_hours", "[]")
            if isinstance(peak_hours, str):
                import ast
                try:
                    peak_hours = ast.literal_eval(peak_hours)
                except Exception:
                    peak_hours = []
            dispatch_time = str(peak_hours[0]) + ":00" if peak_hours else "5:00 PM"

            print("\n  #{} PRIORITY: {}/100".format(rank, row["priority_score"]))
            print("     Location:  {}".format(row["label"]))
            print("     Dispatch:  {}".format(dispatch_time))
            print("     Speed:     -{:.0f}% from free flow".format(row.get("worst_speed_drop_pct", 0)))
            print("     Severity:  {}".format(row.get("impact_severity", "N/A")))
            print("     Violations: {} total".format(row["total_violations"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Enforcement Dispatch CLI")
    parser.add_argument("--top", type=int, default=10, help="Number of top recommendations to show")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    args = parser.parse_args()
    run_cli(top_n=args.top, date=args.date)

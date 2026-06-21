import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import run_pipeline
from src.clustering import run_clustering
from src.traffic_impact import run_impact
from src.scoring import run_scoring
from src.predictive_model import run_prediction
from src.scenario_simulator import run_scenario
from src.analytics import build_adjacency, compute_xai_breakdown, generate_recommendations, compute_early_warning
from config import HEATMAP_DIR
import glob

df = run_pipeline()
df, profiles, ts, clusterer = run_clustering(df)
impact_df, hourly, ripples = run_impact(df, profiles)
total_days = df["date"].nunique()
priority_df, risk_df = run_scoring(profiles, impact_df, total_days)

print("\n--- PREDICTIONS ---")
preds, feature_df, feature_cols, r2 = run_prediction(df, profiles, n_trials=5)
print(f"Predictions: {len(preds)}, R2: {r2:.3f}")

print("\n--- SCENARIO SIMULATOR ---")
scenario = run_scenario(profiles, impact_df, df, vehicle_reduction=50, weather="Heavy Rain", day_type="Festival Day")
print(f"Scenario results: {len(scenario)}")
avg_speed_change = scenario["speed_change"].mean()
print(f"Avg speed change: {avg_speed_change:+.1f} km/h")

print("\n--- XAI ---")
xai = compute_xai_breakdown(df, impact_df, profiles)
print(f"XAI: {len(xai)} hotspots with breakdowns")

print("\n--- RECOMMENDATIONS ---")
recs = generate_recommendations(priority_df, impact_df, risk_df)
print(f"Recommendations: {len(recs)}")

print("\n--- EARLY WARNING ---")
warnings = compute_early_warning(df, profiles)
print(f"Warnings: {len(warnings)} entries")

print("\n--- VISUALIZATIONS ---")
from src.visualization import generate_all_visualizations
generate_all_visualizations(df, profiles, impact_df, priority_df)
maps = sorted(glob.glob(os.path.join(HEATMAP_DIR, "*.html")))
print(f"Maps generated: {len(maps)}")
for m in maps:
    print(f"  {os.path.basename(m)}")

print("\n=== ALL TESTS PASSED! ===")

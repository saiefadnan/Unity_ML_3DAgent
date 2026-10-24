import pandas as pd
import json
from pathlib import Path

# Read just the 2DMultiAgent file
csv_path = Path("allData/test/2DMultiAgent_Test_Results.csv")
df = pd.read_csv(csv_path)

# Get numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

print(f"Processing {csv_path.name}...")
print(f"Shape: {df.shape}")
print(f"\nNumeric columns: {numeric_cols}")

# Build metrics
metrics = {
    "dataset_name": "2DMultiAgent_Test_Results",
    "total_episodes": len(df),
    "total_columns": len(df.columns),
    "numeric_metrics": {}
}

for col in numeric_cols:
    col_data = df[col].dropna()
    if len(col_data) > 0:
        # No outlier filtering for this check
        metrics["numeric_metrics"][col] = {
            "mean": round(float(col_data.mean()), 2),
            "std": round(float(col_data.std()), 2),
            "min": round(float(col_data.min()), 2),
            "max": round(float(col_data.max()), 2),
            "median": round(float(col_data.median()), 2)
        }

# Print results for TeamVictimsRescued
tvr = metrics["numeric_metrics"]["TeamVictimsRescued"]
tv = metrics["numeric_metrics"]["TotalVictims"]

print(f"\nTeamVictimsRescued:")
print(f"  Mean: {tvr['mean']}, Max: {tvr['max']}, Min: {tvr['min']}")
print(f"\nTotalVictims:")
print(f"  Mean: {tv['mean']}, Max: {tv['max']}, Min: {tv['min']}")

if tvr['max'] == tv['max']:
    print('\n✅ FIXED: TeamVictimsRescued max now matches TotalVictims max!')
else:
    print(f'\n❌ ISSUE: TeamVictimsRescued max ({tvr["max"]}) still exceeds TotalVictims max ({tv["max"]})')
    print("\nFull TeamVictimsRescued values:")
    print(df['TeamVictimsRescued'].value_counts().sort_index())

# Save metrics
output_path = Path("output/Individual_Testing_Analysis/2DMultiAgent_Test_Results/metrics.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Saved metrics to {output_path}")

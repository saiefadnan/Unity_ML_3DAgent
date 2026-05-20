import pandas as pd
import json
from pathlib import Path

# Read all metrics.json files
output_folder = Path("output")
test_folders = [
    "Individual_Testing_Analysis/2DAgent_Test_Results",
    "Individual_Testing_Analysis/2DMultiAgent_Test_Results",
    "Individual_Testing_Analysis/3DAgent_Test_Results",
    "Individual_Testing_Analysis/3DMultiAgent_Test_Results"
]

test_summary = []
for folder in test_folders:
    json_file = output_folder / folder / "metrics.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            metrics = json.load(f)
            dataset_name = metrics.get("dataset_name", "Unknown")
            total_episodes = metrics.get("total_episodes", 0)
            
            # Add basic info
            test_row = {
                "Dataset": dataset_name,
                "Total_Episodes": total_episodes,
                "Total_Columns": metrics.get("total_columns", 0)
            }
            
            # Get all numeric metrics
            numeric_metrics = metrics.get("numeric_metrics", {})
            
            print(f"\n{dataset_name}:")
            print(f"  Available metrics: {list(numeric_metrics.keys())}")
            
            # For each metric, add all stats
            for metric_name in ["Episode", "VictimsRescued", "TeamVictimsRescued", "TotalVictims"]:
                if metric_name in numeric_metrics:
                    stat = numeric_metrics[metric_name]
                    test_row[f"{metric_name}_Mean"] = stat["mean"]
                    test_row[f"{metric_name}_Std"] = stat["std"]
                    test_row[f"{metric_name}_Min"] = stat["min"]
                    test_row[f"{metric_name}_Max"] = stat["max"]
                    print(f"  ✅ {metric_name}: {stat['mean']:.2f}")
                else:
                    print(f"  ❌ {metric_name}: NOT FOUND")
                    # Don't add anything - let pandas handle it
            
            test_summary.append(test_row)

# Create DataFrame with all possible columns
all_columns = ["Dataset", "Total_Episodes", "Total_Columns"]
for metric_name in ["Episode", "VictimsRescued", "TeamVictimsRescued", "TotalVictims"]:
    for stat_type in ["Mean", "Std", "Min", "Max"]:
        all_columns.append(f"{metric_name}_{stat_type}")

print("\nCreating DataFrame...")
test_df = pd.DataFrame(test_summary)

# Reindex to include all columns
test_df = test_df.reindex(columns=all_columns)

print("\nBefore filling NaN:")
print(test_df.to_string())

# Format numeric values to 2 decimal places, keep empty as empty
for col in all_columns[3:]:  # Skip first 3 columns
    test_df[col] = test_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '')

print("\nAfter formatting:")
print(test_df.to_string())

# Save
test_csv = output_folder / "MASTER_TEST_STATISTICS.csv"
test_df.to_csv(test_csv, index=False)
print(f"\n✅ Saved: {test_csv}")

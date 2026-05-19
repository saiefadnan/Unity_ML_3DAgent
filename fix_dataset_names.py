"""
Fix dataset names in metrics.json files to match folder names.
This handles cases where folders were renamed after initial extraction.
"""

import json
from pathlib import Path

output_folder = Path("output")

# Find all metrics.json files
metrics_files = list(output_folder.glob("**/metrics.json"))

print("=" * 80)
print("FIXING DATASET NAMES IN METRICS.JSON FILES")
print("=" * 80)

if not metrics_files:
    print("❌ No metrics.json files found!")
else:
    print(f"\n📁 Found {len(metrics_files)} metrics.json files to update\n")
    
    for metrics_path in sorted(metrics_files):
        # Get the parent folder name (which should be the correct dataset name)
        dataset_name = metrics_path.parent.name
        
        # Read the current metrics file
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        old_name = metrics.get("dataset_name", "unknown")
        
        # Update the dataset name
        metrics["dataset_name"] = dataset_name
        
        # Write back
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        status = "✅ UPDATED" if old_name != dataset_name else "✓ UNCHANGED"
        print(f"{status}: {dataset_name}")
        if old_name != dataset_name:
            print(f"         Changed from: '{old_name}'")
            print(f"         Changed to:   '{dataset_name}'")

print("\n" + "=" * 80)
print("✅ DATASET NAME FIXES COMPLETE")
print("=" * 80)

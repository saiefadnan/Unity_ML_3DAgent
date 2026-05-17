"""
Extract episode length trends and generate Figure 2 from training CSV data.
Analyzes how episode completion time changes over training for all setups.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Setup paths
csv_folder = Path("../../csv/env-episode_length")
output_folder = Path(".")
metrics_output = output_folder / "episode_length_metrics.json"
figure_output = output_folder / "Figure2_Episode_Length_Trends.png"

# CSV files and their labels
setups = {
    "S1 2D-Single": "drone7.2_DroneAgent.csv",
    "S2 2D-Multi": "drone6.10_DroneAgent.csv",
    "S3 3D-Single": "drone3d_v15_DroneAgent (1).csv",
    "S4 3D-Multi": "multi_drone3d_v1_DroneAgent (1).csv",
}

# Dictionary to store results
dataframes = {}
metrics_results = {}

print("=" * 70)
print("EXTRACTING EPISODE LENGTH TRENDS FROM CSV FILES")
print("=" * 70)

# Process each setup
for setup_name, csv_filename in setups.items():
    csv_path = csv_folder / csv_filename
    
    if not csv_path.exists():
        print(f"❌ {setup_name}: File not found at {csv_path}")
        continue
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        dataframes[setup_name] = df
        
        # Ensure we have Step and Value columns
        if 'Step' not in df.columns or 'Value' not in df.columns:
            print(f"❌ {setup_name}: Missing 'Step' or 'Value' column")
            print(f"   Available columns: {list(df.columns)}")
            continue
        
        # Calculate statistics
        total_steps = df['Step'].max()
        total_episodes = len(df)
        initial_avg_length = df['Value'].head(100).mean() if len(df) >= 100 else df['Value'].head(10).mean()
        final_avg_length = df['Value'].tail(100).mean() if len(df) >= 100 else df['Value'].tail(10).mean()
        min_length = df['Value'].min()
        max_length = df['Value'].max()
        
        # Calculate trend: reduction in episode length over time
        length_reduction = ((initial_avg_length - final_avg_length) / initial_avg_length * 100) if initial_avg_length > 0 else 0
        
        metrics = {
            "total_data_points": total_episodes,
            "total_training_steps": int(total_steps),
            "initial_avg_episode_length": round(initial_avg_length, 2),
            "final_avg_episode_length": round(final_avg_length, 2),
            "min_episode_length": round(min_length, 2),
            "max_episode_length": round(max_length, 2),
            "episode_length_reduction_%": round(length_reduction, 2),
        }
        
        metrics_results[setup_name] = metrics
        
        print(f"\n✅ {setup_name}")
        print(f"   📊 Total Episodes: {total_episodes}")
        print(f"   🎯 Initial Avg Length: {initial_avg_length:.1f} steps")
        print(f"   🎯 Final Avg Length: {final_avg_length:.1f} steps")
        print(f"   📉 Reduction: {length_reduction:.1f}%")
        print(f"   📈 Min/Max: {min_length:.1f} / {max_length:.1f}")
        
    except Exception as e:
        print(f"❌ {setup_name}: Error - {e}")
        import traceback
        traceback.print_exc()

# Generate Figure 2: Episode Length Trends
print("\n" + "=" * 70)
print("GENERATING FIGURE 2: EPISODE LENGTH TRENDS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(14, 8))

# Define colors and line styles for consistency with Figure 1
colors = {
    "S1 2D-Single": "#1f77b4",      # Blue
    "S2 2D-Multi": "#ff7f0e",       # Orange
    "S3 3D-Single": "#2ca02c",      # Green
    "S4 3D-Multi": "#d62728",       # Red
}

line_styles = {
    "S1 2D-Single": "-",
    "S2 2D-Multi": "--",
    "S3 3D-Single": "-.",
    "S4 3D-Multi": ":",
}

# Plot each setup
for setup_name in ["S1 2D-Single", "S2 2D-Multi", "S3 3D-Single", "S4 3D-Multi"]:
    if setup_name in dataframes:
        df = dataframes[setup_name]
        ax.plot(
            df['Step'],
            df['Value'],
            label=setup_name,
            color=colors[setup_name],
            linestyle=line_styles[setup_name],
            linewidth=2.5,
            alpha=0.8
        )

ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
ax.set_ylabel('Episode Length (steps)', fontsize=13, fontweight='bold')
ax.set_title('Figure 2: Episode Length Trends Across Training Configurations', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.ticklabel_format(style='plain', axis='both')

plt.tight_layout()
plt.savefig(figure_output, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: {figure_output}")

plt.close()

# Save metrics to JSON
print(f"\n💾 Saving metrics to {metrics_output}")
with open(metrics_output, 'w') as f:
    json.dump(metrics_results, f, indent=2)

# Print summary table
print("\n" + "=" * 70)
print("EPISODE LENGTH SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Setup':<20} {'Initial Length':<20} {'Final Length':<20} {'Reduction %':<15}")
print("-" * 75)
for setup in ["S1 2D-Single", "S2 2D-Multi", "S3 3D-Single", "S4 3D-Multi"]:
    if setup in metrics_results:
        m = metrics_results[setup]
        print(f"{setup:<20} {m['initial_avg_episode_length']:<20.2f} {m['final_avg_episode_length']:<20.2f} {m['episode_length_reduction_%']:<15.2f}%")

print("\n" + "=" * 70)
print("✅ EXTRACTION COMPLETE")
print("=" * 70)
print(f"📊 Metrics saved to:  {metrics_output}")
print(f"📈 Figure saved to:   {figure_output}")

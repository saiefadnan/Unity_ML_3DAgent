"""
Extract convergence metrics and generate Figure 1 from training CSV data.
Analyzes reward curves for 2D/3D and Single/Multi-agent setups.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Setup paths
csv_folder = Path("../../csv/env-cumulative_reward")
output_folder = Path(".")
metrics_output = output_folder / "metrics_summary.json"
figure_output = output_folder / "Figure1_Reward_Convergence.png"

# CSV files and their labels
setups = {
    "S1 2D-Single": "drone7.2_DroneAgent.csv",
    "S2 2D-Multi": "drone6.10_DroneAgent.csv",
    "S3 3D-Single": "drone3d_v15_DroneAgent.csv",
    "S4 3D-Multi": "multi_drone3d_v1_DroneAgent.csv",
}

# Dictionary to store results
metrics_results = {}
dataframes = {}

print("=" * 70)
print("EXTRACTING METRICS FROM CSV FILES")
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
        
        # Calculate metrics
        total_steps = df['Step'].max()
        
        # Final average reward: last 10-20% of rows
        tail_percentage = max(int(len(df) * 0.2), 10)  # At least 10 rows
        final_values = df['Value'].tail(tail_percentage)
        final_avg_reward = final_values.mean()
        
        # Find convergence step (where reward plateaus)
        # Method: Calculate rolling std dev and find where it stabilizes
        window_size = max(int(len(df) * 0.05), 50)  # 5% window or min 50 rows
        rolling_std = df['Value'].rolling(window=window_size, min_periods=1).std()
        
        # Convergence threshold: where rolling std is < 5% of final reward's magnitude
        final_reward_magnitude = abs(final_avg_reward)
        if final_reward_magnitude > 0:
            threshold = final_reward_magnitude * 0.05
        else:
            threshold = 1.0
        
        converged_mask = rolling_std < threshold
        
        if converged_mask.any():
            converge_step = df[converged_mask]['Step'].iloc[0]
        else:
            converge_step = total_steps
        
        metrics_results[setup_name] = {
            "Steps to Converge": int(converge_step),
            "Final Average Reward": float(final_avg_reward),
            "Total Steps": int(total_steps),
            "Data Points": len(df),
            "Last 10-20% Tail Size": tail_percentage
        }
        
        print(f"\n✅ {setup_name}")
        print(f"   📊 Steps to Converge:      {converge_step:>12,.0f}")
        print(f"   💰 Final Avg Reward:       {final_avg_reward:>12.2f}")
        print(f"   📈 Total Steps:            {total_steps:>12,.0f}")
        print(f"   📍 Data Points:            {len(df):>12}")
        
    except Exception as e:
        print(f"❌ {setup_name}: Error reading CSV - {e}")

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Setup':<15} {'Steps to Converge':>20} {'Final Avg Reward':>20}")
print("-" * 70)
for setup_name, metrics in metrics_results.items():
    print(f"{setup_name:<15} {metrics['Steps to Converge']:>20,.0f} {metrics['Final Average Reward']:>20.2f}")

# Save metrics to JSON
print(f"\n💾 Saving metrics to {metrics_output}")
with open(metrics_output, 'w') as f:
    json.dump(metrics_results, f, indent=2)

# Generate Figure 2: Reward Convergence Plot
print("\n" + "=" * 70)
print("GENERATING FIGURE 1")
print("=" * 70)

fig, ax = plt.subplots(figsize=(14, 8))

# Color palette for setups
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
ax.set_ylabel('Cumulative Reward', fontsize=13, fontweight='bold')
ax.set_title('Figure 1: Reward Convergence Across Training Configurations', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.ticklabel_format(style='plain', axis='both')

plt.tight_layout()
plt.savefig(figure_output, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: {figure_output}")

plt.close()

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)
print(f"📊 Metrics saved to:  {metrics_output}")
print(f"📈 Figure saved to:   {figure_output}")

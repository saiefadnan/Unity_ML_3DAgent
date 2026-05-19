"""
Individual training analysis for each of the 4 setups.
Similar structure to test data analysis: one folder per setup with detailed visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys

# Setup paths
train_reward_folder = Path("allData/train/env-cumulative_reward").resolve()
train_episode_folder = Path("allData/train/env-episode_length").resolve()
output_folder = Path("output").resolve()
training_detail_folder = output_folder / "05_Individual_Training_Analysis"
training_detail_folder.mkdir(exist_ok=True, parents=True)

print(f"📁 Data paths:")
print(f"   Reward folder: {train_reward_folder}")
print(f"   Episode folder: {train_episode_folder}")
print(f"   Output folder: {output_folder}")
print()

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 90)
print("INDIVIDUAL TRAINING ANALYSIS FOR EACH SETUP")
print("=" * 90)

# Setup names and file mappings
setups = {
    "S1_2D-Single": {
        "train_reward": "2DAgent_Test_Results.csv",
        "train_episode": "2DAgent_Test_Results.csv",
        "name": "S1 2D-Single Agent"
    },
    "S2_2D-Multi": {
        "train_reward": "2DMultiAgent_Train_Results.csv",
        "train_episode": "2DMultiAgent_Train_Results.csv",
        "name": "S2 2D-Multi Agent"
    },
    "S3_3D-Single": {
        "train_reward": "3DAgent_Test_Results.csv",
        "train_episode": "3DAgent_Test_Results.csv",
        "name": "S3 3D-Single Agent"
    },
    "S4_3D-Multi": {
        "train_reward": "3DMultiAgent_Test_Results.csv",
        "train_episode": "3DMultiAgent_Test_Results.csv",
        "name": "S4 3D-Multi Agent"
    }
}

# Verify all files exist before processing
print("\n🔍 Verifying data files...")
for setup_key, setup_info in setups.items():
    reward_path = train_reward_folder / setup_info["train_reward"]
    episode_path = train_episode_folder / setup_info["train_episode"]
    reward_exists = reward_path.exists()
    episode_exists = episode_path.exists()
    print(f"  {setup_key}:")
    print(f"    Reward: {reward_path.name} {'✓' if reward_exists else f'✗ NOT FOUND at {reward_path}'}")
    print(f"    Episode: {episode_path.name} {'✓' if episode_exists else f'✗ NOT FOUND at {episode_path}'}")

colors = {
    "S1_2D-Single": "#1f77b4",
    "S2_2D-Multi": "#ff7f0e",
    "S3_3D-Single": "#2ca02c",
    "S4_3D-Multi": "#d62728"
}

all_training_stats = {}

# ============================================================================
# PROCESS EACH SETUP INDIVIDUALLY
# ============================================================================

for setup_key, setup_info in setups.items():
    setup_name = setup_info["name"]
    print(f"\n{'='*90}")
    print(f"Processing: {setup_name}")
    print(f"{'='*90}")
    
    # Create folder for this setup
    setup_folder = training_detail_folder / setup_key
    setup_folder.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load training data
        reward_path = train_reward_folder / setup_info["train_reward"]
        episode_path = train_episode_folder / setup_info["train_episode"]
        
        reward_df = pd.read_csv(reward_path) if reward_path.exists() else None
        episode_df = pd.read_csv(episode_path) if episode_path.exists() else None
        
        if reward_df is None or episode_df is None:
            print(f"❌ Missing training data files")
            continue
        
        print(f"✓ Loaded {len(reward_df)} reward data points")
        print(f"✓ Loaded {len(episode_df)} episode data points")
        
        # Align datasets to same length (use minimum length)
        min_len = min(len(reward_df), len(episode_df))
        reward_df = reward_df.iloc[:min_len].reset_index(drop=True)
        episode_df = episode_df.iloc[:min_len].reset_index(drop=True)
        
        print(f"✓ Aligned to {min_len} common data points")
        
        # ===== FIGURE 1: DETAILED REWARD ANALYSIS =====
        print(f"\n📊 Creating Figure 1: Reward Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Figure 1: Training Progress - {setup_name}', 
                     fontsize=16, fontweight='bold')
        
        # Subplot (a): Raw reward curve
        ax1 = axes[0, 0]
        ax1.plot(reward_df['Step'], reward_df['Value'], color=colors[setup_key], 
                linewidth=1.5, alpha=0.7, label='Raw Reward')
        ax1.fill_between(reward_df['Step'], reward_df['Value'], alpha=0.2, 
                        color=colors[setup_key])
        ax1.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Cumulative Reward', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Raw Reward Curve', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Subplot (b): Smoothed reward (rolling average)
        ax2 = axes[0, 1]
        window = max(len(reward_df) // 50, 10)
        smoothed = reward_df['Value'].rolling(window=window, center=True).mean()
        ax2.plot(reward_df['Step'], smoothed, color=colors[setup_key], 
                linewidth=3, label=f'Smoothed (window={window})', marker='o', markersize=4)
        ax2.fill_between(reward_df['Step'], smoothed, alpha=0.2, color=colors[setup_key])
        ax2.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Reward (Smoothed)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Smoothed Reward Trend', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Subplot (c): Reward distribution
        ax3 = axes[1, 0]
        ax3.hist(reward_df['Value'], bins=50, color=colors[setup_key], 
                alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.axvline(reward_df['Value'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {reward_df["Value"].mean():.2f}')
        ax3.axvline(reward_df['Value'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {reward_df["Value"].median():.2f}')
        ax3.set_xlabel('Reward Value', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('(c) Reward Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Subplot (d): Cumulative improvement
        ax4 = axes[1, 1]
        # Calculate improvement: first 10% vs last 10%
        first_10_pct = int(len(reward_df) * 0.1)
        last_10_pct = int(len(reward_df) * 0.1)
        
        first_avg = reward_df['Value'].head(first_10_pct).mean()
        last_avg = reward_df['Value'].tail(last_10_pct).mean()
        improvement = ((last_avg - first_avg) / abs(first_avg)) * 100 if first_avg != 0 else 0
        
        improvement_pct = [((reward_df['Value'].iloc[:i].mean() - first_avg) / abs(first_avg)) * 100 
                          for i in range(1, len(reward_df) + 1)]
        
        # Use indices (0 to len-1) for x-axis to match improvement_pct length
        x_indices = range(len(improvement_pct))
        
        ax4.plot(x_indices, improvement_pct, color=colors[setup_key], 
                linewidth=2.5, marker='o', markersize=3, label='Improvement %')
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax4.fill_between(x_indices, improvement_pct, 0, 
                        where=[x >= 0 for x in improvement_pct], alpha=0.2, color='green')
        ax4.fill_between(x_indices, improvement_pct, 0, 
                        where=[x < 0 for x in improvement_pct], alpha=0.2, color='red')
        ax4.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax4.set_title('(d) Cumulative Improvement', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        fig.savefig(setup_folder / "Figure1_reward_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: Figure1_reward_analysis.png")
        
        # ===== FIGURE 2: EPISODE LENGTH ANALYSIS =====
        print(f"\n📊 Creating Figure 2: Episode Length Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Figure 2: Episode Length Trends - {setup_name}', 
                     fontsize=16, fontweight='bold')
        
        # Subplot (a): Raw episode length
        ax1 = axes[0, 0]
        ax1.plot(episode_df['Step'], episode_df['Value'], color=colors[setup_key], 
                linewidth=1.5, alpha=0.7, label='Raw Episode Length')
        ax1.fill_between(episode_df['Step'], episode_df['Value'], alpha=0.2, 
                        color=colors[setup_key])
        ax1.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Episode Length (steps)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Raw Episode Length', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Subplot (b): Smoothed episode length
        ax2 = axes[0, 1]
        window = max(len(episode_df) // 50, 10)
        smoothed_ep = episode_df['Value'].rolling(window=window, center=True).mean()
        ax2.plot(episode_df['Step'], smoothed_ep, color=colors[setup_key], 
                linewidth=3, label=f'Smoothed (window={window})', marker='o', markersize=4)
        ax2.fill_between(episode_df['Step'], smoothed_ep, alpha=0.2, color=colors[setup_key])
        ax2.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Episode Length (smoothed)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Smoothed Episode Length', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Subplot (c): Episode length distribution
        ax3 = axes[1, 0]
        ax3.hist(episode_df['Value'], bins=50, color=colors[setup_key], 
                alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.axvline(episode_df['Value'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {episode_df["Value"].mean():.0f}')
        ax3.axvline(episode_df['Value'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {episode_df["Value"].median():.0f}')
        ax3.set_xlabel('Episode Length (steps)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('(c) Episode Length Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Subplot (d): Episode length reduction
        ax4 = axes[1, 1]
        first_avg_ep = episode_df['Value'].head(first_10_pct).mean()
        last_avg_ep = episode_df['Value'].tail(last_10_pct).mean()
        reduction_pct = ((first_avg_ep - last_avg_ep) / first_avg_ep) * 100 if first_avg_ep != 0 else 0
        
        reduction_trend = [((episode_df['Value'].head(i).mean() - episode_df['Value'].iloc[i-1]) / 
                          episode_df['Value'].head(i).mean()) * 100 
                          for i in range(1, len(episode_df))]
        
        # Use indices (0 to len-2) for x-axis to match reduction_trend length
        x_indices_ep = range(len(reduction_trend))
        
        ax4.plot(x_indices_ep, reduction_trend, color=colors[setup_key], 
                linewidth=2.5, marker='o', markersize=3, label='Reduction %')
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax4.fill_between(x_indices_ep, reduction_trend, 0, 
                        where=[x >= 0 for x in reduction_trend], alpha=0.2, color='green')
        ax4.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Efficiency Gain (%)', fontsize=11, fontweight='bold')
        ax4.set_title('(d) Episode Length Reduction', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        fig.savefig(setup_folder / "Figure2_episode_length_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: Figure2_episode_length_analysis.png")
        
        # ===== CALCULATE STATISTICS =====
        print(f"\n📋 Calculating training statistics...")
        
        stats = {
            "setup_name": setup_name,
            "folder_name": setup_folder.name,  # Use actual folder name for consistency
            "total_training_steps": int(reward_df['Step'].max()),
            "total_episodes": len(reward_df),
            "reward": {
                "initial_avg": float(reward_df['Value'].head(first_10_pct).mean()),
                "final_avg": float(reward_df['Value'].tail(last_10_pct).mean()),
                "max": float(reward_df['Value'].max()),
                "min": float(reward_df['Value'].min()),
                "mean": float(reward_df['Value'].mean()),
                "std": float(reward_df['Value'].std()),
                "improvement_pct": float(improvement)
            },
            "episode_length": {
                "initial_avg": float(episode_df['Value'].head(first_10_pct).mean()),
                "final_avg": float(episode_df['Value'].tail(last_10_pct).mean()),
                "max": float(episode_df['Value'].max()),
                "min": float(episode_df['Value'].min()),
                "mean": float(episode_df['Value'].mean()),
                "std": float(episode_df['Value'].std()),
                "reduction_pct": float(reduction_pct)
            }
        }
        
        all_training_stats[setup_key] = stats
        
        # ===== SAVE STATISTICS =====
        stats_csv_path = setup_folder / "training_statistics.csv"
        stats_df = pd.DataFrame({
            "Metric": [
                "Total Training Steps",
                "Total Episodes",
                "Reward: Initial Avg",
                "Reward: Final Avg",
                "Reward: Improvement %",
                "Reward: Max",
                "Reward: Mean",
                "Reward: Std Dev",
                "Episode Length: Initial Avg",
                "Episode Length: Final Avg",
                "Episode Length: Reduction %",
                "Episode Length: Max",
                "Episode Length: Mean",
                "Episode Length: Std Dev"
            ],
            "Value": [
                stats["total_training_steps"],
                stats["total_episodes"],
                f'{stats["reward"]["initial_avg"]:.2f}',
                f'{stats["reward"]["final_avg"]:.2f}',
                f'{stats["reward"]["improvement_pct"]:.2f}',
                f'{stats["reward"]["max"]:.2f}',
                f'{stats["reward"]["mean"]:.2f}',
                f'{stats["reward"]["std"]:.2f}',
                f'{stats["episode_length"]["initial_avg"]:.0f}',
                f'{stats["episode_length"]["final_avg"]:.0f}',
                f'{stats["episode_length"]["reduction_pct"]:.2f}',
                f'{stats["episode_length"]["max"]:.0f}',
                f'{stats["episode_length"]["mean"]:.0f}',
                f'{stats["episode_length"]["std"]:.0f}'
            ]
        })
        
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"   ✅ Saved: training_statistics.csv")
        
        # Save JSON
        stats_json_path = setup_folder / "training_statistics.json"
        with open(stats_json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   ✅ Saved: training_statistics.json")
        
        print(f"\n📊 Training Summary for {setup_name}:")
        print(f"   Reward improvement: {stats['reward']['improvement_pct']:.1f}%")
        print(f"   Episode length reduction: {stats['episode_length']['reduction_pct']:.1f}%")
        
    except Exception as e:
        print(f"❌ Error processing {setup_name}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print(f"\n{'='*90}")
print("GENERATING COMPARISON TABLE")
print(f"{'='*90}")

comparison_data = []
for setup_key, stats in all_training_stats.items():
    comparison_data.append({
        "Setup": stats["setup_name"],
        "Total Steps": stats["total_training_steps"],
        "Reward Improvement (%)": f'{stats["reward"]["improvement_pct"]:.1f}',
        "Initial Episode Length": f'{stats["episode_length"]["initial_avg"]:.0f}',
        "Final Episode Length": f'{stats["episode_length"]["final_avg"]:.0f}',
        "Episode Reduction (%)": f'{stats["episode_length"]["reduction_pct"]:.1f}'
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_path = training_detail_folder / "training_comparison_table.csv"
comparison_df.to_csv(comparison_path, index=False)

print(f"\n{comparison_df.to_string(index=False)}")
print(f"\n✅ Saved: training_comparison_table.csv")

# Save comparison as JSON
comparison_json = training_detail_folder / "all_training_statistics.json"
with open(comparison_json, 'w') as f:
    json.dump(all_training_stats, f, indent=2)
print(f"✅ Saved: all_training_statistics.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*90}")
print("✅ INDIVIDUAL TRAINING ANALYSIS COMPLETE")
print(f"{'='*90}")

print(f"""
📁 OUTPUT STRUCTURE:

output/05_Individual_Training_Analysis/
├── S1_2D-Single/
│   ├── Figure1_reward_analysis.png              (4 reward subplots)
│   ├── Figure2_episode_length_analysis.png      (4 episode subplots)
│   ├── training_statistics.csv
│   └── training_statistics.json
│
├── S2_2D-Multi/
│   ├── Figure1_reward_analysis.png
│   ├── Figure2_episode_length_analysis.png
│   ├── training_statistics.csv
│   └── training_statistics.json
│
├── S3_3D-Single/
│   ├── Figure1_reward_analysis.png
│   ├── Figure2_episode_length_analysis.png
│   ├── training_statistics.csv
│   └── training_statistics.json
│
├── S4_3D-Multi/
│   ├── Figure1_reward_analysis.png
│   ├── Figure2_episode_length_analysis.png
│   ├── training_statistics.csv
│   └── training_statistics.json
│
├── training_comparison_table.csv          (Compare all 4 setups)
└── all_training_statistics.json           (All metrics in JSON)

📊 CONTENT:

Each setup includes:
- Figure 1: Reward analysis (raw, smoothed, distribution, improvement)
- Figure 2: Episode length analysis (raw, smoothed, distribution, reduction)
- Statistics CSV: 14 key metrics per setup
- Statistics JSON: Machine-readable detailed data

🎓 Ready for thesis - detailed training analysis for each configuration!

{'='*90}
""")

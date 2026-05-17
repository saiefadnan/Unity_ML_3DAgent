"""
Extract all 13 metrics from test run CSV files and generate Table V.
Analyzes evaluation episodes across all 4 setups.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Setup paths
csv_folder = Path("../../csv/test_run")
output_folder = Path(".")
metrics_output = output_folder / "table_v_metrics.json"
table_output = output_folder / "Table_V_Results.txt"

# CSV files and their labels
setups = {
    "S1 2D-Single": ("SingleAgent2D_Test_Results.csv", "single"),
    "S2 2D-Multi": ("MultiAgent2D_Test_Results.csv", "multi"),
    "S3 3D-Single": ("SingleAgent3D_Test_Results.csv", "single"),
    "S4 3D-Multi": ("MultiAgent3D_Test_Results.csv", "multi"),
}

# Dictionary to store results
metrics_results = {}

print("=" * 80)
print("EXTRACTING TABLE V METRICS FROM TEST RUN CSVs")
print("=" * 80)

# Process each setup
for setup_name, (csv_filename, agent_type) in setups.items():
    csv_path = csv_folder / csv_filename
    
    if not csv_path.exists():
        print(f"❌ {setup_name}: File not found at {csv_path}")
        continue
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"\n✅ {setup_name} ({agent_type})")
        print(f"   Rows: {len(df)}")
        
        # Show EndReason values for debugging
        if 'EndReason' in df.columns:
            print(f"   EndReason values: {df['EndReason'].unique().tolist()}")
        
        metrics = {}
        
        # ===== METRIC 1: Success Rate (%) =====
        # Success = VictimsRescued == TotalVictims
        if agent_type == "single":
            if 'VictimsRescued' in df.columns and 'TotalVictims' in df.columns:
                success_mask = df['VictimsRescued'] == df['TotalVictims']
                success_episodes = success_mask.sum()
                success_rate = (success_episodes / len(df)) * 100
                metrics['Success Rate (%)'] = round(success_rate, 2)
                print(f"   ✓ Success Rate: {success_rate:.2f}% ({success_episodes}/{len(df)})")
        else:  # multi
            if 'TeamVictimsRescued' in df.columns and 'TotalVictims' in df.columns:
                success_mask = df['TeamVictimsRescued'] == df['TotalVictims']
                success_episodes = success_mask.sum()
                success_rate = (success_episodes / len(df)) * 100
                metrics['Success Rate (%)'] = round(success_rate, 2)
                print(f"   ✓ Success Rate: {success_rate:.2f}% ({success_episodes}/{len(df)})")
        
        # ===== METRIC 2: Avg Victims / Episode =====
        if agent_type == "single":
            if 'VictimsRescued' in df.columns:
                avg_victims = df['VictimsRescued'].mean()
                metrics['Avg Victims / Episode'] = round(avg_victims, 2)
                print(f"   ✓ Avg Victims/Episode: {avg_victims:.2f}")
        else:  # multi
            if 'TeamVictimsRescued' in df.columns:
                avg_victims = df['TeamVictimsRescued'].mean()
                metrics['Avg Victims / Episode'] = round(avg_victims, 2)
                print(f"   ✓ Avg Victims/Episode: {avg_victims:.2f}")
        
        # ===== METRIC 3: Avg Steps to Complete =====
        # AVERAGEIF(EndReason, "*all_victims_rescued*", StepsTaken) OR similar completion criteria
        if 'StepsTaken' in df.columns and 'EndReason' in df.columns:
            # Look for "Completion" or "team_completion" or any success-indicating EndReason
            completion_mask = df['EndReason'].str.contains(
                'Completion|all_victims_rescued|success', 
                case=False, na=False
            )
            completion_df = df[completion_mask]
            if len(completion_df) > 0:
                avg_steps = completion_df['StepsTaken'].mean()
                metrics['Avg Steps to Complete'] = round(avg_steps, 0)
                print(f"   ✓ Avg Steps to Complete (EndReason completion): {avg_steps:.0f}")
            else:
                print(f"   ⚠ No completion episodes found in EndReason")
        
        # ===== METRIC 4: Completion Time (s) =====
        if 'Avg Steps to Complete' in metrics and metrics['Avg Steps to Complete'] > 0:
            completion_time = metrics['Avg Steps to Complete'] * 0.02
            metrics['Completion Time (s)'] = round(completion_time, 2)
            print(f"   ✓ Completion Time: {completion_time:.2f}s")
        
        # ===== METRIC 5: Avg HP Survival =====
        # AVERAGE(DroneHP column) - all episodes
        if agent_type == "single":
            if 'DroneHP' in df.columns:
                avg_hp = df['DroneHP'].mean()
                metrics['Avg HP Survival'] = round(avg_hp, 2)
                print(f"   ✓ Avg HP Survival: {avg_hp:.2f}")
        else:  # multi
            # Average HP across all agents (all episodes)
            hp_cols = [col for col in df.columns if 'HP' in col and col.startswith('Agent')]
            if hp_cols:
                avg_hp = df[hp_cols].mean().mean()
                metrics['Avg HP Survival'] = round(avg_hp, 2)
                print(f"   ✓ Avg HP Survival (all agents): {avg_hp:.2f}")
        
        # ===== METRIC 6: Crash Rate (%) =====
        # COUNTIF(EndReason, "*crash*") / total rows × 100
        if 'EndReason' in df.columns:
            crash_mask = df['EndReason'].str.contains('crash', case=False, na=False)
            crash_episodes = crash_mask.sum()
            crash_rate = (crash_episodes / len(df)) * 100
            metrics['Crash Rate (%)'] = round(crash_rate, 2)
            print(f"   ✓ Crash Rate: {crash_rate:.2f}%")
        
        # ===== METRIC 7: Collision Count =====
        # Look for collision-related columns
        collision_sum = 0
        found_collision = False
        for col in df.columns:
            if 'collision' in col.lower():
                collision_sum += df[col].sum()
                found_collision = True
                print(f"   ✓ Collision Count from '{col}': {collision_sum}")
                break
        
        if found_collision:
            metrics['Collision Count'] = int(collision_sum)
        
        # ===== METRIC 8: Stability (uprightness) =====
        # Look for stability-related columns
        for col in df.columns:
            if 'stability' in col.lower() or 'upright' in col.lower():
                stability = df[col].mean()
                metrics['Stability (uprightness)'] = round(stability, 3)
                print(f"   ✓ Stability: {stability:.3f}")
                break
        
        # ===== METRIC 9: Avg Distance Traveled (m) =====
        if agent_type == "single":
            if 'DistanceTraveled' in df.columns:
                avg_distance = df['DistanceTraveled'].mean()
                metrics['Avg Distance Traveled (m)'] = round(avg_distance, 2)
                print(f"   ✓ Avg Distance Traveled: {avg_distance:.2f}m")
        else:  # multi
            # Sum distances for all agents if available
            dist_cols = [col for col in df.columns if 'Distance' in col or 'distance' in col]
            if dist_cols:
                total_distance = df[dist_cols].sum(axis=1).mean()
                metrics['Avg Distance Traveled (m)'] = round(total_distance, 2)
                print(f"   ✓ Avg Distance Traveled (team): {total_distance:.2f}m")
        
        # ===== METRIC 10: Path Efficiency =====
        if agent_type == "single":
            if 'PathEfficiency' in df.columns:
                # Filter out NaN and invalid values
                valid_efficiency = df['PathEfficiency'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_efficiency) > 0:
                    avg_efficiency = valid_efficiency.mean()
                    # Cap extremely high values (likely calculation errors)
                    if avg_efficiency < 100:
                        metrics['Path Efficiency'] = round(avg_efficiency, 3)
                        print(f"   ✓ Path Efficiency: {avg_efficiency:.3f}")
                    else:
                        # Use median instead for robustness
                        median_efficiency = valid_efficiency.median()
                        metrics['Path Efficiency'] = round(median_efficiency, 3)
                        print(f"   ✓ Path Efficiency (median): {median_efficiency:.3f}")
        else:  # multi
            # Average efficiency across all agents
            eff_cols = [col for col in df.columns if 'Efficiency' in col]
            if eff_cols:
                # Filter invalid values
                all_efficiencies = pd.concat([df[col].replace([np.inf, -np.inf], np.nan) for col in eff_cols])
                valid_eff = all_efficiencies.dropna()
                if len(valid_eff) > 0:
                    median_eff = valid_eff.median()
                    metrics['Path Efficiency'] = round(median_eff, 3)
                    print(f"   ✓ Path Efficiency (team median): {median_eff:.3f}")
        
        # ===== METRIC 11: Area Coverage (cells) - Single agent only =====
        if agent_type == "single":
            for col in df.columns:
                if 'explored' in col.lower() or 'coverage' in col.lower() or col == 'ExploredCells':
                    avg_cells = df[col].mean()
                    metrics['Area Coverage (cells)'] = round(avg_cells, 0)
                    print(f"   ✓ Area Coverage: {avg_cells:.0f} cells")
                    break
        
        # ===== METRIC 12: Team Coverage (cells) - Multi agent only =====
        if agent_type == "multi":
            for col in df.columns:
                if 'explored' in col.lower() or 'coverage' in col.lower() or 'TeamExplored' in col:
                    avg_cells = df[col].mean()
                    metrics['Team Coverage (cells)'] = round(avg_cells, 0)
                    print(f"   ✓ Team Coverage: {avg_cells:.0f} cells")
                    break
        
        metrics_results[setup_name] = metrics

        metrics_results[setup_name] = metrics
        
    except Exception as e:
        print(f"❌ {setup_name}: Error - {e}")
        import traceback
        traceback.print_exc()

# Print summary table
print("\n" + "=" * 80)
print("TABLE V: EVALUATION RESULTS")
print("=" * 80)

# Prepare table data
metrics_list = [
    'Success Rate (%)',
    'Avg Victims / Episode',
    'Avg Steps to Complete',
    'Completion Time (s)',
    'Avg HP Survival',
    'Crash Rate (%)',
    'Collision Count',
    'Stability (uprightness)',
    'Avg Distance Traveled (m)',
    'Path Efficiency',
    'Area Coverage (cells)',
    'Team Coverage (cells)',
]

# Print formatted table
print(f"\n{'Metric':<35} {'S1 2D-Single':>15} {'S2 2D-Multi':>15} {'S3 3D-Single':>15} {'S4 3D-Multi':>15}")
print("-" * 95)

for metric in metrics_list:
    row = f"{metric:<35}"
    for setup in ["S1 2D-Single", "S2 2D-Multi", "S3 3D-Single", "S4 3D-Multi"]:
        if setup in metrics_results and metric in metrics_results[setup]:
            value = metrics_results[setup][metric]
            # Handle N/A cases
            if metric == 'Area Coverage (cells)' and 'Multi' in setup:
                row += f"{'N/A':>15}"
            elif metric == 'Team Coverage (cells)' and 'Single' in setup:
                row += f"{'N/A':>15}"
            else:
                row += f"{str(value):>15}"
        else:
            row += f"{'—':>15}"
    print(row)

# Save to JSON
print(f"\n💾 Saving metrics to {metrics_output}")
with open(metrics_output, 'w') as f:
    json.dump(metrics_results, f, indent=2)

# Save table to text file
print(f"📊 Saving table to {table_output}")
with open(table_output, 'w') as f:
    f.write("TABLE V: EVALUATION RESULTS ACROSS 4 SETUPS\n")
    f.write("=" * 95 + "\n\n")
    f.write(f"{'Metric':<35} {'S1 2D-Single':>15} {'S2 2D-Multi':>15} {'S3 3D-Single':>15} {'S4 3D-Multi':>15}\n")
    f.write("-" * 95 + "\n")
    for metric in metrics_list:
        row = f"{metric:<35}"
        for setup in ["S1 2D-Single", "S2 2D-Multi", "S3 3D-Single", "S4 3D-Multi"]:
            if setup in metrics_results and metric in metrics_results[setup]:
                value = metrics_results[setup][metric]
                if metric == 'Area Coverage (cells)' and 'Multi' in setup:
                    row += f"{'N/A':>15}"
                elif metric == 'Team Coverage (cells)' and 'Single' in setup:
                    row += f"{'N/A':>15}"
                else:
                    row += f"{str(value):>15}"
            else:
                row += f"{'—':>15}"
        f.write(row + "\n")

print("\n" + "=" * 80)
print("✅ EXTRACTION COMPLETE")
print("=" * 80)

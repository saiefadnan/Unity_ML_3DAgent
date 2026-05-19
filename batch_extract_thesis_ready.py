"""
Enhanced batch extraction with thesis-ready visualizations.
Organizes each dataset in separate folders with no redundancy.
Adds enhanced labels, figure numbers, and professional formatting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys

# Setup paths
alldata_folder = Path("allData/test")
output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

# Set matplotlib style for thesis
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

print("=" * 80)
print("THESIS-READY BATCH DATA EXTRACTION")
print("Organizing outputs by dataset with enhanced visualizations")
print("=" * 80)

# Get all CSV files in allData
csv_files = sorted(list(alldata_folder.glob("*.csv")))
if not csv_files:
    print(f"❌ No CSV files found in {alldata_folder}")
    sys.exit(1)

print(f"\n📁 Found {len(csv_files)} CSV files in {alldata_folder}")

all_dataframes = {}
all_metrics = {}

# ============================================================================
# PROCESS EACH CSV FILE
# ============================================================================

for csv_idx, csv_path in enumerate(csv_files, 1):
    filename = csv_path.name
    setup_name = filename.replace(".csv", "")
    
    # Create separate folder for each setup
    setup_folder = output_folder / setup_name
    setup_folder.mkdir(exist_ok=True, parents=True)
    
    # Use the folder name as dataset_name (important for renamed folders)
    # This ensures that if a folder is renamed after creation, the metrics reflect the new name
    dataset_display_name = setup_folder.name
    
    print(f"\n{'='*80}")
    print(f"[Setup {csv_idx}/{len(csv_files)}] {setup_name}")
    print(f"{'='*80}")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        all_dataframes[setup_name] = df
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate statistics for metrics file
        # Use dataset_display_name (the actual folder name) instead of CSV filename
        # This ensures renamed folders get the correct name in metrics.json
        metrics = {
            "dataset_name": dataset_display_name,
            "total_episodes": len(df),
            "total_columns": len(df.columns),
            "numeric_metrics": {}
        }
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                metrics["numeric_metrics"][col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median())
                }
        
        all_metrics[setup_name] = metrics
        
        # ===== PLOT 1: INDIVIDUAL METRICS WITH SUBPLOTS =====
        print(f"\n📊 Generating individual metrics plot...")
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        axes = axes.flatten()
        
        fig.suptitle(f'Figure 1: Individual Metric Distributions - {dataset_display_name}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        subplot_labels = [chr(97 + i) for i in range(len(numeric_cols))]
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            col_data = df[col].dropna()
            x_vals = range(len(col_data))
            
            # Plot with gradient fill
            ax.fill_between(x_vals, col_data, alpha=0.25, color='steelblue')
            ax.plot(x_vals, col_data, color='steelblue', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
            
            # Enhanced title with subplot label
            ax.set_title(f'({subplot_labels[idx]}) {col}', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Episode Index', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics box
            stats_text = f'μ: {col_data.mean():.2f}\nσ: {col_data.std():.2f}\nMin: {col_data.min():.2f}\nMax: {col_data.max():.2f}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.5),
                   family='monospace')
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        individual_path = setup_folder / "Figure1_individual_metrics.png"
        plt.tight_layout()
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {individual_path.relative_to(output_folder)}")
        
        # ===== PLOT 2: NORMALIZED OVERLAY (SINGLE COMPREHENSIVE PLOT) =====
        print(f"📊 Generating normalized overlay plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(numeric_cols)))
        
        for col_idx, col in enumerate(numeric_cols):
            col_data = df[col].dropna()
            # Normalize to 0-1 range
            if col_data.max() - col_data.min() != 0:
                normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
            else:
                normalized = col_data
            
            x_vals = range(len(normalized))
            ax.plot(x_vals, normalized, label=col, alpha=0.8, linewidth=2.5, 
                   color=colors[col_idx], marker='o', markersize=3)
        
        ax.set_xlabel('Episode Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Value (0-1)', fontsize=12, fontweight='bold')
        ax.set_title(f'Figure 2: Normalized Metric Trends - {dataset_display_name}', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='best', ncol=2, framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)
        
        # Add axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        normalized_path = setup_folder / "Figure2_normalized_metrics.png"
        plt.tight_layout()
        plt.savefig(normalized_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {normalized_path.relative_to(output_folder)}")
        
        # ===== PLOT 3: LOG SCALE (ALWAYS GENERATE FOR CONSISTENCY) =====
        print(f"📊 Generating log-scale plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for col_idx, col in enumerate(numeric_cols):
            col_data = df[col].dropna()
            col_data_positive = col_data[col_data > 0]
            
            if len(col_data_positive) > 0:
                x_vals = range(len(col_data_positive))
                ax.semilogy(x_vals, col_data_positive, label=col, alpha=0.8, 
                          linewidth=2.5, color=colors[col_idx], marker='o', markersize=3)
        
        ax.set_xlabel('Episode Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value (Log Scale)', fontsize=12, fontweight='bold')
        ax.set_title(f'Figure 3: Logarithmic Scale Metrics - {dataset_display_name}', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='best', ncol=2, framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # Add axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        logscale_path = setup_folder / "Figure3_logscale_metrics.png"
        plt.tight_layout()
        plt.savefig(logscale_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {logscale_path.relative_to(output_folder)}")
        
        # ===== SAVE STATISTICS CSV =====
        print(f"📋 Generating statistics table...")
        
        summary_df = pd.DataFrame({
            col: [
                metrics["numeric_metrics"][col]["mean"],
                metrics["numeric_metrics"][col]["std"],
                metrics["numeric_metrics"][col]["min"],
                metrics["numeric_metrics"][col]["max"],
                metrics["numeric_metrics"][col]["median"]
            ]
            for col in numeric_cols
        }, index=['Mean', 'Std Dev', 'Min', 'Max', 'Median'])
        
        summary_path = setup_folder / "metrics_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"   ✅ Saved: {summary_path.relative_to(output_folder)}")
        
        # ===== SAVE METRICS JSON =====
        metrics_path = setup_folder / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   ✅ Saved: {metrics_path.relative_to(output_folder)}")
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# CROSS-DATASET COMPARISON PLOT
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING CROSS-DATASET COMPARISON")
print(f"{'='*80}")

if len(all_dataframes) > 1:
    # Find common numeric columns
    common_numeric = set(all_dataframes[list(all_dataframes.keys())[0]].select_dtypes(include=[np.number]).columns)
    for df in all_dataframes.values():
        common_numeric &= set(df.select_dtypes(include=[np.number]).columns)
    
    common_numeric = sorted(list(common_numeric))[:6]  # Limit to 6 for clarity
    
    if common_numeric:
        print(f"\n🔄 Creating cross-dataset comparison for {len(common_numeric)} common metrics...")
        
        n_cols = 3
        n_rows = (len(common_numeric) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        axes = axes.flatten()
        
        fig.suptitle('Figure 4: Cross-Dataset Performance Comparison', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Color palette for datasets
        cmap = plt.cm.get_cmap('tab10')
        dataset_colors = [cmap(i) for i in np.linspace(0, 1, len(all_dataframes))]
        
        subplot_labels = [chr(97 + i) for i in range(len(common_numeric))]
        
        for plot_idx, col in enumerate(common_numeric):
            ax = axes[plot_idx]
            
            for dataset_idx, (setup_name, df) in enumerate(all_dataframes.items()):
                col_data = df[col].dropna()
                if col_data.max() - col_data.min() != 0:
                    normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                else:
                    normalized = col_data
                
                x_vals = range(len(normalized))
                ax.plot(x_vals, normalized, label=setup_name, alpha=0.8, linewidth=2.5,
                       color=dataset_colors[dataset_idx], marker='o', markersize=3)
            
            ax.set_title(f'({subplot_labels[plot_idx]}) {col}', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Episode Index', fontsize=10)
            ax.set_ylabel('Normalized Value (0-1)', fontsize=10)
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(-0.05, 1.05)
        
        # Hide unused subplots
        for idx in range(len(common_numeric), len(axes)):
            axes[idx].axis('off')
        
        comparison_path = output_folder / "Figure4_cross_dataset_comparison.png"
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {comparison_path.relative_to(output_folder)}")

# ============================================================================
# GENERATE FIGURE CAPTIONS DOCUMENT
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING FIGURE CAPTIONS DOCUMENT")
print(f"{'='*80}")

captions_doc = """
# THESIS FIGURE CAPTIONS

## Figure 1: Individual Metric Distributions
Each subplot (a-*) displays a single metric from the test results across all episodes.
The y-axis represents the metric value, while the x-axis shows episode progression.
The shaded area beneath each curve helps visualize trends. The inset statistics box 
shows mean (μ), standard deviation (σ), minimum, and maximum values for each metric.
This visualization is useful for understanding the behavior and variance of individual 
performance indicators.

## Figure 2: Normalized Metric Trends
All metrics are normalized to a 0-1 scale to enable direct visual comparison despite 
having different units and ranges. This allows identification of correlated behaviors 
and divergent patterns across metrics. The normalized representation facilitates 
understanding of relative changes and helps identify convergence or degradation patterns 
during testing.

## Figure 3: Logarithmic Scale Metrics
For datasets containing metrics with extreme value ranges (e.g., path efficiency ranging 
from 0.8 to 566K), logarithmic scaling is employed. This preserves the visibility of 
both small and large values. The log-scale representation is essential for metrics where 
multiplicative rather than additive differences are meaningful, such as efficiency ratios.

## Figure 4: Cross-Dataset Performance Comparison
This figure compares the same metrics across different datasets (e.g., 2D single-agent, 
2D multi-agent, 3D single-agent, 3D multi-agent). Each metric is normalized independently 
for fair comparison. Different colors represent different datasets, enabling visual 
assessment of relative performance and identifying which configurations achieve superior 
outcomes for specific metrics.

---

## Dataset Statistics Summary

"""

for setup_name in sorted(all_metrics.keys()):
    metrics = all_metrics[setup_name]
    captions_doc += f"\n### {setup_name}\n"
    captions_doc += f"- **Total Episodes**: {metrics['total_episodes']}\n"
    captions_doc += f"- **Metrics Tracked**: {metrics['total_columns']}\n"
    captions_doc += f"- **Numeric Metrics**: {len(metrics['numeric_metrics'])}\n\n"
    
    captions_doc += "**Key Statistics**:\n"
    for metric_name, stats in sorted(metrics['numeric_metrics'].items())[:5]:
        captions_doc += f"- {metric_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
        captions_doc += f"range=[{stats['min']:.2f}, {stats['max']:.2f}]\n"
    
    if len(metrics['numeric_metrics']) > 5:
        captions_doc += f"- ... and {len(metrics['numeric_metrics']) - 5} more metrics\n"

captions_path = output_folder / "FIGURE_CAPTIONS.md"
with open(captions_path, 'w') as f:
    f.write(captions_doc)

print(f"✅ Saved: {captions_path.relative_to(output_folder)}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print(f"\n{'='*80}")
print("✅ THESIS-READY EXTRACTION COMPLETE")
print(f"{'='*80}")

print(f"\n📁 Output Structure:")
print(f"   output/")
for setup_name in sorted(all_metrics.keys()):
    print(f"   ├── {setup_name}/")
    print(f"   │   ├── Figure1_individual_metrics.png")
    print(f"   │   ├── Figure2_normalized_metrics.png")
    print(f"   │   ├── Figure3_logscale_metrics.png (if applicable)")
    print(f"   │   ├── metrics_summary.csv")
    print(f"   │   └── metrics.json")

print(f"   ├── Figure4_cross_dataset_comparison.png")
print(f"   └── FIGURE_CAPTIONS.md")

print(f"\n📊 Ready-to-Use Figures:")
print(f"   - Figure 1: Individual metrics with subplots (a, b, c, ...)")
print(f"   - Figure 2: Normalized overlay for trend comparison")
print(f"   - Figure 3: Log-scale for extreme value ranges")
print(f"   - Figure 4: Cross-dataset performance comparison")

print(f"\n💾 Supporting Files:")
print(f"   - metrics_summary.csv: Statistics table for each dataset")
print(f"   - metrics.json: Machine-readable metrics data")
print(f"   - FIGURE_CAPTIONS.md: Professional figure descriptions")

print(f"\n{'='*80}")

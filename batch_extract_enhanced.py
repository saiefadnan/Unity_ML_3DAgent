"""
Enhanced batch extraction with better multi-scale visualizations.
Uses subplots and normalized scaling to show all metrics clearly.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys

# Setup paths
alldata_folder = Path("allData")
output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

print("=" * 80)
print("ENHANCED BATCH DATA EXTRACTION WITH IMPROVED VISUALIZATIONS")
print("=" * 80)

# Get all CSV files in allData
csv_files = list(alldata_folder.glob("*.csv"))
if not csv_files:
    print(f"❌ No CSV files found in {alldata_folder}")
    sys.exit(1)

print(f"\n📁 Found {len(csv_files)} CSV files in {alldata_folder}")

all_dataframes = {}

# ============================================================================
# PROCESS EACH CSV FILE WITH IMPROVED PLOTS
# ============================================================================

for csv_path in sorted(csv_files):
    filename = csv_path.name
    print(f"\n{'='*80}")
    print(f"Processing: {filename}")
    print(f"{'='*80}")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        setup_name = filename.replace(".csv", "")
        all_dataframes[setup_name] = df
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # ===== PLOT 1: INDIVIDUAL SUBPLOTS (NO SCALE ISSUES) =====
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
            axes = axes.flatten()  # Flatten to 1D array
            fig.suptitle(f"All Metrics (Individual Scales): {setup_name}", 
                        fontsize=16, fontweight='bold', y=1.00)
            
            for idx, col in enumerate(numeric_cols):
                ax = axes[idx]
                col_data = df[col].dropna()
                x_vals = range(len(col_data))
                
                # Plot with color gradient
                ax.fill_between(x_vals, col_data, alpha=0.3, color='steelblue')
                ax.plot(x_vals, col_data, color='steelblue', linewidth=2, marker='o', markersize=3)
                
                ax.set_title(f"{col}", fontsize=11, fontweight='bold')
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
                
                # Show stats on plot
                stats_text = f"Mean: {col_data.mean():.2f}\nMin: {col_data.min():.2f}\nMax: {col_data.max():.2f}"
                ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Hide unused subplots
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].axis('off')
            
            individual_path = output_folder / f"{setup_name}_individual_metrics.png"
            plt.tight_layout()
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {individual_path}")
            
            # ===== PLOT 2: NORMALIZED OVERLAY (0-1 SCALE) =====
            fig, ax = plt.subplots(figsize=(14, 7))
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                # Normalize to 0-1 range
                if col_data.max() - col_data.min() != 0:
                    normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                else:
                    normalized = col_data
                
                x_vals = range(len(normalized))
                ax.plot(x_vals, normalized, label=col, alpha=0.8, linewidth=2)
            
            ax.set_xlabel("Sample Index", fontsize=12, fontweight='bold')
            ax.set_ylabel("Normalized Value (0-1)", fontsize=12, fontweight='bold')
            ax.set_title(f"All Metrics Normalized to 0-1 Scale: {setup_name}", 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            
            normalized_path = output_folder / f"{setup_name}_normalized_all_metrics.png"
            plt.tight_layout()
            plt.savefig(normalized_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {normalized_path}")
            
            # ===== PLOT 3: LOG SCALE (FOR LARGE VALUE RANGES) =====
            if any(df[col].max() > 100 for col in numeric_cols):
                fig, ax = plt.subplots(figsize=(14, 7))
                
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    # Only plot non-zero values for log scale
                    col_data_positive = col_data[col_data > 0]
                    if len(col_data_positive) > 0:
                        x_vals = range(len(col_data_positive))
                        ax.semilogy(x_vals, col_data_positive, label=col, alpha=0.8, linewidth=2, marker='o', markersize=4)
                
                ax.set_xlabel("Sample Index", fontsize=12, fontweight='bold')
                ax.set_ylabel("Value (Log Scale)", fontsize=12, fontweight='bold')
                ax.set_title(f"All Metrics (Log Scale) for Large Value Ranges: {setup_name}", 
                            fontsize=14, fontweight='bold')
                ax.legend(fontsize=10, loc='best', ncol=2)
                ax.grid(True, alpha=0.3, which='both')
                
                logscale_path = output_folder / f"{setup_name}_logscale_metrics.png"
                plt.tight_layout()
                plt.savefig(logscale_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Saved: {logscale_path}")
            
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# GENERATE COMPARISON PLOTS
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING COMPARISON PLOTS ACROSS DATASETS")
print(f"{'='*80}")

if len(all_dataframes) > 1:
    # Find common numeric columns across all dataframes
    common_numeric = set(all_dataframes[list(all_dataframes.keys())[0]].select_dtypes(include=[np.number]).columns)
    for df in all_dataframes.values():
        common_numeric &= set(df.select_dtypes(include=[np.number]).columns)
    
    common_numeric = sorted(list(common_numeric))
    
    if common_numeric:
        # Plot first 6 common columns
        n_cols = min(3, len(common_numeric))
        n_rows = (min(6, len(common_numeric)) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        axes = axes.flatten()
        fig.suptitle("Cross-Dataset Comparison", fontsize=16, fontweight='bold')
        
        for plot_idx, col in enumerate(common_numeric[:6]):
            ax = axes[plot_idx]
            
            # Normalize each dataset separately for fair comparison
            for setup_name, df in all_dataframes.items():
                col_data = df[col].dropna()
                if col_data.max() - col_data.min() != 0:
                    normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                else:
                    normalized = col_data
                
                x_vals = range(len(normalized))
                ax.plot(x_vals, normalized, label=setup_name, alpha=0.8, linewidth=2)
            
            ax.set_title(f"{col} (Normalized)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Normalized Value (0-1)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(common_numeric[:6]), len(axes)):
            axes[idx].axis('off')
        
        comparison_path = output_folder / "cross_dataset_comparison_normalized.png"
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {comparison_path}")

print(f"\n{'='*80}")
print("✅ ENHANCED EXTRACTION COMPLETE")
print(f"{'='*80}")
print(f"\n📁 Output folder: {output_folder.resolve()}")
print(f"📊 Generated visualizations:")
print(f"   - *_individual_metrics.png (each metric in separate subplot)")
print(f"   - *_normalized_all_metrics.png (all metrics scaled 0-1)")
print(f"   - *_logscale_metrics.png (large value ranges with log scale)")
print(f"   - cross_dataset_comparison_normalized.png (compare datasets)")
print(f"\n💡 TIP: Different plots show different aspects:")
print(f"   - Individual: Best for seeing metric ranges and patterns")
print(f"   - Normalized: Best for comparing multiple metrics together")
print(f"   - Log Scale: Best for metrics with huge value ranges")
print(f"\n{'='*80}")

# Python Analysis & Extraction Tools

This directory contains the Python analysis pipeline used to parse training TensorBoard logs, test evaluation CSVs, and generate thesis-ready figures and LaTeX tables.

## Execution

All scripts should be executed from the **repository root directory** (`Unity_ML_3DAgent`):

```bash
# Activate your Python virtual environment
venv\Scripts\activate

# 1. Run the comprehensive cross-setup analysis (combines training & test metrics)
python scripts/analysis_scripts/comprehensive_analysis.py

# 2. Extract thesis-ready figures and distributions
python scripts/analysis_scripts/batch_extract_thesis_ready.py

# 3. Generate IEEE Table V and Table VI formatted data
python scripts/analysis_scripts/generate_tables_v_vi.py

# 4. Generate rankings and LaTeX tables
python scripts/analysis_scripts/generate_thesis_metrics.py

# 5. Run individual training curve analysis
python scripts/analysis_scripts/individual_training_analysis.py
```

## Script Overview

| Script | Purpose | Output Location |
|---|---|---|
| `comprehensive_analysis.py` | Full cross-experiment evaluation combining S1, S2, S3, and S4 data | `output/01_Training_Analysis/`, `output/02_Test_Analysis/`, `output/03_Cross_Analysis/` |
| `batch_extract_thesis_ready.py` | Generates box plots, metric histograms, and figure captions | `output/Individual_Testing_Analysis/` |
| `generate_tables_v_vi.py` | Formats Table V (all 13 metrics) and Table VI (baseline comparison) | `output/baseline_analysis/` |
| `generate_thesis_metrics.py` | Calculates stability rankings, correlations, and LaTeX tables | `output/` |
| `individual_training_analysis.py` | Detailed per-model training curve and step analysis | `output/Individual_Training_Analysis/` |
| `regenerate_metrics_2d_multi.py` | Recalculates team statistics for 2D multi-agent evaluations | `output/Individual_Testing_Analysis/2DMultiAgent_Test_Results/` |
| `debug_test_stats.py` | Quick CLI debugger for generated metric JSON files | Console output |
| `verify_fix.py` | Verifies data parsing integrity for multi-agent metric rows | Console output |

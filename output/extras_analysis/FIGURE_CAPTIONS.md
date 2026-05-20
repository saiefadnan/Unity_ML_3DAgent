
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


### 2DAgent_Test_Results
- **Total Episodes**: 100
- **Metrics Tracked**: 9
- **Numeric Metrics**: 8

**Key Statistics**:
- DistanceTraveled: mean=139.16, std=19.33, range=[106.02, 192.71]
- DroneHP: mean=-2.94, std=47.14, range=[-270.10, 100.00]
- Episode: mean=50.50, std=29.01, range=[1.00, 100.00]
- ExploredCells: mean=12.53, std=5.31, range=[3.00, 25.00]
- PathEfficiency: mean=0.25, std=0.06, range=[0.15, 0.39]
- ... and 3 more metrics

### 2DMultiAgent_Test_Results
- **Total Episodes**: 100
- **Metrics Tracked**: 15
- **Numeric Metrics**: 14

**Key Statistics**:
- Agent0Efficiency: mean=1.82, std=0.79, range=[0.59, 3.67]
- Agent0Goals: mean=2.04, std=0.53, range=[1.00, 3.00]
- Agent0HP: mean=93.44, std=11.94, range=[38.60, 100.00]
- Agent1Efficiency: mean=2.04, std=0.93, range=[0.66, 4.97]
- Agent1Goals: mean=1.47, std=0.58, range=[0.00, 3.00]
- ... and 9 more metrics

### 3DAgent_Test_Results
- **Total Episodes**: 100
- **Metrics Tracked**: 8
- **Numeric Metrics**: 7

**Key Statistics**:
- DistanceTraveled: mean=232.38, std=155.85, range=[0.00, 502.77]
- DroneHP: mean=267.97, std=214.08, range=[-32.80, 500.00]
- Episode: mean=50.50, std=29.01, range=[1.00, 100.00]
- PathEfficiency: mean=98337.83, std=175543.20, range=[0.81, 566672.40]
- StepsTaken: mean=2652.25, std=1889.41, range=[0.00, 5000.00]
- ... and 2 more metrics

### 3DMultiAgent_Test_Results
- **Total Episodes**: 100
- **Metrics Tracked**: 15
- **Numeric Metrics**: 14

**Key Statistics**:
- Agent0Efficiency: mean=1.08, std=0.81, range=[0.00, 3.43]
- Agent0Goals: mean=3.61, std=2.07, range=[0.00, 8.00]
- Agent0HP: mean=379.27, std=132.29, range=[-31.40, 500.00]
- Agent1Efficiency: mean=29.54, std=275.34, range=[0.48, 2755.35]
- Agent1Goals: mean=3.61, std=2.07, range=[0.00, 10.00]
- ... and 9 more metrics

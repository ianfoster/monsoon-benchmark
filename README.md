# Monsoon Onset Benchmarking Framework

A decision-oriented benchmarking framework for evaluating AI weather prediction (AIWP) models on Indian monsoon onset forecasting.

## Overview

This framework implements the methodology described in "Decision-oriented benchmarking to transform AI weather forecast access: Application to the Indian monsoon." It enables systematic evaluation of weather forecast models for predicting monsoon onset—a critical decision point for agricultural planning in India.

### Key Features

- **Onset Detection**: Modified Moron-Robertson index (rainfall-based) and Webster-Yang Index (circulation-based)
- **Deterministic Metrics**: Mean Absolute Error (MAE), False Alarm Rate (FAR), Miss Rate (MR)
- **Probabilistic Metrics**: Fair Brier Score, Fair Ranked Probability Score, AUC with ensemble size adjustment
- **Spatial Analysis**: 4°×4° grid evaluation over India with Core Monsoon Zone (CMZ) focus
- **Visualization**: Skill maps, reliability diagrams, ROC curves, time series plots

## Installation

```bash
pip install -e .
```

### Dependencies

Core dependencies:
- numpy, xarray, pandas, scipy
- matplotlib, cartopy (visualization)
- scikit-learn (metrics)

Optional for data access:
- cdsapi (ERA5 download)
- xesmf (conservative regridding)

## Quick Start

```python
from monsoon_benchmark.indices import compute_local_onset, compute_wyi
from monsoon_benchmark.metrics import compute_mae, compute_fair_brier_score
from monsoon_benchmark.evaluation import MonsoonBenchmark

# Load your data
# imd_rainfall: xr.DataArray with daily precipitation
# u200, u850: xr.DataArray with zonal winds

# Compute onset using Modified Moron-Robertson index
onset_result = compute_local_onset(
    precip=imd_rainfall,
    wet_spell_threshold=50.0,  # mm, climatological value
    wet_day_threshold_mm=1.0,
    wet_spell_days=5,
    mok_date="06-02",  # Search starts after June 2
)

# Compute Webster-Yang circulation index
wyi = compute_wyi(u200, u850)

# Evaluate forecasts
mae = compute_mae(forecast_onsets, observed_onsets)
```

## Project Structure

```
monsoon_benchmark/
├── data/           # Data loaders (IMD, ERA5) and regridding
├── indices/        # Onset index computation
├── metrics/        # Deterministic and probabilistic metrics
├── models/         # AIWP model wrappers
├── evaluation/     # Benchmarking framework
└── visualization/  # Plotting utilities
```

## Example Notebooks

1. **01_data_exploration.ipynb** - Load and visualize IMD rainfall and ERA5 data
2. **02_onset_index_validation.ipynb** - Compute and validate onset indices
3. **03_deterministic_evaluation.ipynb** - Evaluate with MAE, FAR, Miss Rate
4. **04_probabilistic_evaluation.ipynb** - Evaluate ensembles with BSS, RPSS, AUC
5. **05_2025_case_study.ipynb** - Real-time application example

## Methodology

### Onset Definition

Local monsoon onset is defined as the first day of the first 5-day wet spell after the Monsoon Onset over Kerala (MOK) median date (June 2), where:
- Each day has precipitation ≥ 1 mm/day
- Total accumulation exceeds the local climatological 5-day wet spell amount

### Evaluation Grid

Models are evaluated on a 4°×4° grid over India (6-38°N, 66-98°E), with particular focus on the Core Monsoon Zone (18-28°N, 74-86°E).

### Fair Skill Scores

Probabilistic metrics use the Ferro et al. adjustment for ensemble size:

```
Fair Brier Score = BS - (1/M) * p * (1-p)
```

where M is the ensemble size and p is the forecast probability.

## Configuration

Default parameters in `configs/default.yaml`:

```yaml
onset:
  mok_median_date: "06-02"
  wet_day_threshold_mm: 1.0
  wet_spell_days: 5

evaluation:
  forecast_windows:
    medium_range: [1, 15]
    subseasonal: [16, 30]
  tolerance_days:
    medium_range: 3
    subseasonal: 5

grid:
  resolution_deg: 4.0
```

## Data Sources

- **IMD**: India Meteorological Department 1° gridded daily rainfall (1901-present)
- **ERA5**: ECMWF reanalysis for u-wind at 200 hPa and 850 hPa

## Reference

This implementation is based on:

> Masiwal et al., "Decision-oriented benchmarking to transform AI weather forecast access: Application to the Indian monsoon"
>
> arXiv: [https://arxiv.org/abs/2602.03767](https://arxiv.org/abs/2602.03767)

## License

MIT License

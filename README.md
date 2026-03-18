# HF Factor ZOO

Replication of paper [The High-Frequency Factor Zoo](https://www.sakethaleti.com/research).

## Replication Pipeline

The analysis has three main branches: **intraday market prediction**, **overnight prediction**, and **panel (stock-level) prediction**. All branches share the same data download step.

### Step 1: Download Data

**`prepare_aleti.R`**

Downloads intraday (1-minute) factor returns and Fama-French factors from Google Drive. Saves raw data to `data/` and `data/factor_returns/`.

Requires: Google Drive authentication via `googledrive`.

### Step 2: Construct Features and Targets

Three scripts prepare the data for the three estimation branches:

| Script | Input | Output | Description |
|---|---|---|---|
| **`factors_zoo.R`** | `data/factor_returns/`, `data/ff6_1min_returns.csv` | `data/factor_returns.csv` | Aggregates 1-min factor returns to 15-min. Creates rolling features (1h, 1d windows). Runs jump detection (Bollerslev-Todorov). Constructs intraday market return targets (total, continuous, jump). |
| **`factors_zoo_overnight.R`** | `data/factor_returns/`, `data/ff6_1min_returns.csv` | `data/factor_returns_overnight.csv` | Creates features at multiple horizons (5m to 1d). Target is the next-day overnight (open) return. |
| **`estimation_panel.R`** (data section) | `data/factor_returns.csv`, `data/ohlcv_15min.csv` | (in-memory) | Merges market factors with individual stock 15-min OHLCV data. Target is per-stock next-period return. |

### Step 3: Estimation (ML Benchmark)

Each estimation script defines autotuned ML models and creates a `batchtools` registry for HPC execution:

| Script | Data | Registry folder | Models |
|---|---|---|---|
| **`estimation.R`** | `data/factor_returns.csv` | `experiments/` | RF, XGBoost, NNET, BART, NN (torch), EARTH, GBM, LightGBM. Each model has a standard and a jump-adjusted variant. |
| **`estimation_fi.R`** | `data/factor_returns.csv` | `experiments_fi/` | Same models + Cubist, CatBoost, AORSF. Adds feature selection (JMIM, mRMR, CMIM, importance, CAR score). |
| **`estimation_overnight.R`** | `data/factor_returns_overnight.csv` | `experiments_overnight/` | RF, XGBoost, NNET, BART, NN, EARTH, GBM, LightGBM. |
| **`estimation_panel.R`** | `data/factor_returns.csv` + `data/ohlcv_15min.csv` | `experiments_panel/` | RF, XGBoost, NNET, BART, NN, EARTH, GBM, LightGBM, Cubist. |

All use Gap-CV (expanding window) resampling and Hyperband tuning.

**Running on HPC (Padobran):**
1. Run the estimation script locally (non-interactive) to create the registry.
2. Submit jobs via the generated PBS script (`padobran_estimation.sh`, `padobran_estimation_fi.sh`, etc.).
3. Each array job runs `run_job.R`, which loads the registry and executes one experiment.

**Running locally (test):**
Run the estimation script interactively -- it creates a small `experiments_test/` registry and submits a single job.

### Step 4: Extract Results

**`extraction.R`**

Reads completed experiments from `experiments_fi/`, extracts predictions and tuning results, and saves:
- `experiments_fi/predictions.csv` -- out-of-sample predictions per model and fold
- `experiments_fi/results.rds` -- inner tuning results
- `experiments_fi/fi.csv` -- selected features per fold

### Step 5: Strategy Backtesting

| Script | Experiments folder | Description |
|---|---|---|
| **`strategy.R`** | `experiments_fi/` | Backtest intraday strategy on SPY. Applies sign-based trading rule with transaction cost adjustment. Computes Sharpe ratios, equity curves. Uploads predictions to Azure/QuantConnect. |
| **`strategy_overnight.R`** | `experiments_overnight/` | Same logic for overnight predictions. |
| **`strategy_panel.R`** | `experiments_panel/` | Same logic for stock-level panel predictions. |

### Step 6: Analyze Results

**`results.R`** -- Reads feature importance data from `experiments_fi/fi.csv` and ranks features by frequency.

## Other Scripts (not part of main pipeline)

| Script | Purpose |
|---|---|
| `prepare.R` | Prepares minute-level stock data from Databento. Samples symbols, upsamples to 5-min. Used for panel data setup. |
| `predictors_padobran.R` | Computes predictors on HPC (per-stock). |
| `stocks_minute.R` | Downloads/processes minute-level stock data. |
| `stocks_minute_alphapicks.R` | Variant of above for AlphaPicks universe. |
| `run_job.R` | HPC job runner -- loads batchtools registry and executes one job by PBS array index. |
| `temp.R` | Scratch/temporary analysis. |

## Infrastructure

- **HPC**: Jobs run on Padobran cluster via PBS. Apptainer containers (`image.sif`, `image_predictors.sif`) provide the R environment.
- **Container definitions**: `image.def`, `image_est.def`, `image_predictors.def`.
- **Shell scripts**: `padobran_*.sh` (PBS job arrays), `extraction.sh`, `image.sh`, `predictors_padobran.sh`.

## Key R Packages

- `mlr3` ecosystem (mlr3, mlr3pipelines, mlr3tuning, mlr3batchmark, mlr3finance, mlr3torch)
- `batchtools` -- HPC job management
- `data.table`, `arrow` -- data manipulation
- `highfrequency` -- realized volatility and jump detection
- `PerformanceAnalytics` -- strategy performance metrics

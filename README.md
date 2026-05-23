# Comparative Time Series Analysis and Forecasting of Mobile Network Traffic

**Formative Assignment 1 — Machine Learning Techniques I**

Analysis of the Telecom Italia Milan CDR dataset covering 10,000 grid squares across a two-month period. The project addresses three tasks: memory-efficient large-scale data handling, exploratory time series and spatial analysis, and one-step-ahead internet traffic forecasting using SARIMA, LSTM, and CNN-LSTM models.

---

## Project Structure

```
Telecom/
├── data/                        # Raw daily .txt files (~5 GB — not committed)
├── processed/                   # Pipeline outputs — generated, not committed
│   ├── cell_totals.csv          #   total 2-month traffic per grid square
│   ├── target_areas.csv         #   the three focal areas
│   ├── area_timeseries.parquet  #   10-min series for the three target areas
│   ├── city_heatmap.csv         #   grid-structured totals for spatial plot
│   └── all_results.pkl          #   model results (written by notebook 3)
├── tables/                      # Quality/summary tables — generated, not committed
│   └── parse_quality_log.csv    #   per-file data quality counts
├── notebooks/
│   ├── 00_introduction.ipynb    # Problem overview and dataset description
│   ├── 01_data_handling.ipynb   # Task 1: memory-efficient loading and preprocessing
│   ├── 02_eda.ipynb             # Task 2: exploratory data analysis
│   ├── 03_model_training.ipynb     # Task 3: SARIMA, LSTM, CNN-LSTM models
│   └── 04_conclusion.ipynb      # Key findings summary
├── src/
│   ├── config.py                # All configurable parameters (dates, paths, hyperparameters)
│   └── preprocessing.py         # Data parsing, validation, and processing module
├── pipeline.py                  # End-to-end data pipeline — runs without Jupyter
├── milano-grid.geojson          # Milan 100×100 grid geometry (WGS84)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requirements

### Python version
Python **3.10 or later** is required. The project was developed and tested on Python 3.11.

### Hardware
- **RAM**: 8 GB minimum. 16 GB recommended for processing all 62 files without chunking issues.
- **Disk**: ~6 GB free for the raw data, plus ~200 MB for processed outputs.
- **GPU**: Not required. TensorFlow will use CPU by default. Training times on CPU are reported in notebook 3.

### Packages
All dependencies are pinned in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `statsmodels` | SARIMA model, ADF test, decomposition, ACF/PACF |
| `pmdarima` | Automated SARIMA parameter search (`auto_arima`) |
| `tensorflow` | LSTM and CNN-LSTM models |
| `scikit-learn` | MinMaxScaler, error metrics |
| `geopandas` | Spatial heatmap from GeoJSON |
| `pyarrow` | Parquet read/write |
| `matplotlib`, `seaborn` | Visualisation |
| `psutil` | Hardware info for Task 1 report |

---

## Setup & Reproduction

### Step 1 — Download the dataset

Download all 62 daily files from Harvard Dataverse:

> **Telecommunications activity (Milan)**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

Place all `.txt` files in the `data/` directory. Files are named:

```
sms-call-internet-mi-2013-11-01.txt
sms-call-internet-mi-2013-11-02.txt
...
sms-call-internet-mi-2013-12-31.txt
```

The grid geometry (`milano-grid.geojson`) is already included in the repository.
It is also available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

> **Minimum for Task 3**: files from November 1 through December 22 (~52 files).
> The January file is not needed since the test window ends December 22.

### Step 2 — Process the raw data

**Option A — command line (recommended, faster):**

```bash
python pipeline.py
```

This reads all files in `data/`, validates each row, aggregates internet traffic
across country codes, and writes the processed outputs to `processed/` and `tables/`.
Use `--force` to reprocess even if outputs already exist:

```bash
python pipeline.py --force
```

**Option B — via Jupyter notebook:**

```bash
jupyter notebook notebooks/
```

Open `01_data_handling.ipynb` and run all cells. This produces the same outputs
as Option A while also showing the memory optimisation comparison.

**Expected runtime** (approximate, on an 8-core consumer CPU):

| Step | Time |
|------|------|
| Processing 1 file (~300 MB) | 15–30 s |
| Processing all 62 files | 15–30 min |
| Loading from Parquet in later notebooks | < 5 s |

### Step 3 — Run the analysis notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks in order. Each notebook saves its outputs for the next one.

| Notebook | Task | Input | Key Output |
|----------|------|-------|------------|
| `00_introduction.ipynb` | Overview | — | — |
| `01_data_handling.ipynb` | Task 1 | `data/*.txt` | `processed/` directory |
| `02_eda.ipynb` | Task 2 | `processed/` | 7+ analysis figures |
| `03_model_training.ipynb` | Task 3 | `processed/` | 9 prediction plots, metric tables |
| `04_conclusion.ipynb` | Summary | `processed/all_results.pkl` | Consolidated results |

---

## Configuration

All tunable parameters are in **`src/config.py`**. Edit that file to change any
parameter — no notebook changes are needed.

```python
# Train / test split
TRAIN_START = pd.Timestamp("2013-11-01")
TRAIN_END   = pd.Timestamp("2013-12-15 23:50")
TEST_START  = pd.Timestamp("2013-12-16")      # held-out test week
TEST_END    = pd.Timestamp("2013-12-22 23:50")

# Sequence length for neural models (1 day = 144 × 10-min slots)
SEQ_LEN = 144

# Neural network hyperparameters (final, post-tuning)
LSTM_UNITS   = [64, 32]
CNN_FILTERS  = [32, 64]
DROPOUT_RATE = 0.2
BATCH_SIZE   = 32
MAX_EPOCHS   = 50
```

---

## Models

Three one-step-ahead forecasting models are implemented in `notebooks/03_model_training.ipynb`:

| Model | Type | Description |
|-------|------|-------------|
| **SARIMA** | Statistical | Seasonal ARIMA with daily period (s=144). Parameters selected via `auto_arima`. Rolling one-step-ahead forecast. |
| **LSTM** | Neural network | Two-layer stacked LSTM (64→32 units) with dropout. Input: 144 past values. |
| **CNN-LSTM** | Neural network | Two Conv1D layers (32→64 filters) + MaxPooling + LSTM (50 units) + dropout. Extracts local patterns before sequential modelling. |

All models are trained and evaluated independently on each of three target areas:
- **Area A**: grid square with the highest total internet traffic over the two-month period
- **Area B**: square 4159
- **Area C**: square 4556

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'config'`**
Each notebook begins with a sys.path cell that adds `src/` to the import path.
Make sure you are running Jupyter from the project root or the `notebooks/` directory,
not from inside `src/`.

**`FileNotFoundError: No data files found`**
The `data/` directory exists but contains no `.txt` files. Download the dataset
from the Harvard Dataverse link above.

**`ValueError: Target squares not found in data`**
Square IDs 4159 or 4556 are missing because not all files were downloaded.
These squares may have zero activity in the files you have — download the
full 62-file set to resolve this.

**geopandas / Fiona install errors on Windows**
Install the binary wheel directly:
```bash
pip install geopandas --prefer-binary
```

**TensorFlow warning about CUDA / GPU**
These are informational only. The models train on CPU by default. To suppress:
```python
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
```

---

## Data Citation

G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi,
F. Antonelli, A. Vespignani, A. Pentland, and B. Lepri,
"A multi-source dataset of urban life in the city of Milan and the Province of Trentino,"
*Scientific Data*, vol. 2, p. 150055, Sep. 2015.
doi:[10.1038/sdata.2015.55](https://doi.org/10.1038/sdata.2015.55)

Dataset released under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).

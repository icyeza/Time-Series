"""Central configuration for the Telecom Milan traffic analysis project."""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives in src/ — project root is one level up
PROJECT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_DIR / "data"
PROCESSED_DIR = PROJECT_DIR / "processed"
TABLES_DIR    = PROJECT_DIR / "tables"
GRID_PATH     = PROJECT_DIR / "milano-grid.geojson"

# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------
GRID_SIZE         = 100          # 100 × 100 squares
N_SQUARES         = GRID_SIZE ** 2
SLOT_MINUTES      = 10           # each time interval is 10 minutes
SLOTS_PER_DAY     = 24 * 60 // SLOT_MINUTES   # 144
SLOTS_PER_WEEK    = SLOTS_PER_DAY * 7         # 1008
DATA_FREQUENCY    = "10min"

# ---------------------------------------------------------------------------
# Train / test split  (adjust here — nowhere else)
# ---------------------------------------------------------------------------
TRAIN_START = pd.Timestamp("2013-11-01")
TRAIN_END   = pd.Timestamp("2013-12-15 23:50")
TEST_START  = pd.Timestamp("2013-12-16")
TEST_END    = pd.Timestamp("2013-12-22 23:50")

# ---------------------------------------------------------------------------
# Target areas (from assignment spec)
# ---------------------------------------------------------------------------
FIXED_SQUARE_IDS = [4159, 4556]   # Area B and C; Area A is determined from data

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
CHUNKSIZE = 200_000   # rows per chunk when reading raw files

# ---------------------------------------------------------------------------
# Neural network hyperparameters (final, after tuning)
# ---------------------------------------------------------------------------
SEQ_LEN       = SLOTS_PER_DAY   # 144 — one full day of history
LSTM_UNITS    = [64, 32]
CNN_FILTERS   = [32, 64]
CNN_KERNEL    = 3
DROPOUT_RATE  = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE    = 32
MAX_EPOCHS    = 50
ES_PATIENCE   = 5

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
SEED = 42

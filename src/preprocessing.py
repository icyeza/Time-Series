"""
Telecom Italia Milan dataset — parsing and preprocessing utilities.

Reads the raw tab-separated daily files, validates each field, aggregates
internet traffic across country codes, and builds the processed outputs used
by the analysis and forecasting notebooks.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schema (corrected field order — see assignment note re: paper error)
# ---------------------------------------------------------------------------
_RAW_COLUMN_NAMES = [
    "square_id",
    "time_interval",   # Unix milliseconds
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_traffic",
]

# Only these three are needed for the assignment
KEEP_COLUMNS = ["square_id", "time_interval", "internet_traffic"]


# ---------------------------------------------------------------------------
# Data quality tracking
# ---------------------------------------------------------------------------
@dataclass
class ParseReport:
    """Per-file data quality summary produced during chunked reading."""

    filename: str
    rows_parsed: int = 0
    blank_traffic_values: int = 0       # NaN or empty string — treated as 0
    bad_square_ids: int = 0             # non-numeric or out of range
    bad_timestamps: int = 0             # non-numeric
    bad_traffic_values: int = 0         # non-numeric and non-blank
    country_code_rows_collapsed: int = 0  # duplicate (square, time) rows merged

    @property
    def total_issues(self) -> int:
        return (
            self.bad_square_ids
            + self.bad_timestamps
            + self.bad_traffic_values
        )

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def find_raw_files(directory: Path) -> list[Path]:
    """Return daily telecom files sorted chronologically."""
    return sorted(directory.glob("sms-call-internet-mi-*.txt"))


# ---------------------------------------------------------------------------
# Chunked reader
# ---------------------------------------------------------------------------
def _parse_chunk(raw: pd.DataFrame, report: ParseReport) -> pd.DataFrame:
    """
    Validate and type-convert one raw string chunk.

    Strategy: read everything as string, then coerce to numeric so that
    malformed values can be counted and skipped rather than silently cast.
    """
    report.rows_parsed += len(raw)

    # Blank internet_traffic → legitimate zero-activity interval
    is_blank = raw["internet_traffic"].isna() | (raw["internet_traffic"].str.strip() == "")
    report.blank_traffic_values += int(is_blank.sum())

    sq  = pd.to_numeric(raw["square_id"],       errors="coerce")
    ts  = pd.to_numeric(raw["time_interval"],   errors="coerce")
    act = pd.to_numeric(raw["internet_traffic"], errors="coerce")

    bad_sq  = sq.isna()
    bad_ts  = ts.isna()
    bad_act = act.isna() & ~is_blank   # NaN that was NOT originally blank

    report.bad_square_ids     += int(bad_sq.sum())
    report.bad_timestamps     += int(bad_ts.sum())
    report.bad_traffic_values += int(bad_act.sum())

    keep = ~(bad_sq | bad_ts | bad_act)
    out = pd.DataFrame({
        "square_id":        sq[keep].astype("uint16"),
        "time_interval":    ts[keep].astype("int64"),
        "internet_traffic": act[keep].fillna(0.0).astype("float32"),
    })
    return out


def iter_file_chunks(
    file_path: Path,
    chunksize: int,
) -> tuple[pd.DataFrame, ParseReport]:
    """
    Yield (aggregated_chunk, report) for each chunk of one daily file.

    Each yielded DataFrame has one row per (square_id, time_interval) —
    internet_traffic is summed across country codes within the chunk.
    The report accumulates across all chunks; the final yield has complete totals.
    """
    report = ParseReport(filename=file_path.name)
    reader = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=_RAW_COLUMN_NAMES,
        usecols=KEEP_COLUMNS,
        chunksize=chunksize,
        dtype="string",
        on_bad_lines="skip",
        engine="c",
    )
    for raw_chunk in reader:
        chunk = _parse_chunk(raw_chunk, report)
        if chunk.empty:
            yield chunk, report
            continue

        dupes = int(chunk.duplicated(["square_id", "time_interval"]).sum())
        report.country_code_rows_collapsed += dupes

        aggregated = (
            chunk
            .groupby(["square_id", "time_interval"], as_index=False, observed=True)
            ["internet_traffic"]
            .sum()
        )
        yield aggregated, report


# ---------------------------------------------------------------------------
# Naive load  (for memory baseline comparison in notebook 1)
# ---------------------------------------------------------------------------
def load_file_naive(file_path: Path) -> pd.DataFrame:
    """Load all 8 columns with default dtypes — used only for the memory demo."""
    return pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=_RAW_COLUMN_NAMES,
    )


# ---------------------------------------------------------------------------
# Cell totals  (all 10,000 squares, full period)
# ---------------------------------------------------------------------------
def compute_cell_totals(
    files: list[Path],
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sum internet traffic per grid square across the entire dataset.

    Returns
    -------
    cell_totals : DataFrame with columns [square_id, total_traffic]
    quality_log : DataFrame with one row per file, from ParseReport
    """
    running_totals: dict[int, float] = {}
    reports: list[ParseReport] = []
    active_report: ParseReport | None = None

    for fp in files:
        for chunk, rpt in iter_file_chunks(fp, chunksize):
            active_report = rpt
            if chunk.empty:
                continue
            for sq, val in chunk.groupby("square_id", observed=True)["internet_traffic"].sum().items():
                running_totals[int(sq)] = running_totals.get(int(sq), 0.0) + float(val)
        if active_report is not None:
            reports.append(active_report)
            active_report = None

    totals_df = pd.DataFrame(
        sorted(running_totals.items()),
        columns=["square_id", "total_traffic"],
    )
    totals_df["square_id"]    = totals_df["square_id"].astype("uint16")
    totals_df["total_traffic"] = totals_df["total_traffic"].astype("float64")

    quality_df = pd.DataFrame([r.as_dict() for r in reports])
    return totals_df, quality_df


# ---------------------------------------------------------------------------
# Target area selection
# ---------------------------------------------------------------------------
def identify_target_areas(
    cell_totals: pd.DataFrame,
    fixed_ids: list[int],
) -> pd.DataFrame:
    """
    Return the three target areas: highest-traffic + the two fixed square IDs.

    Parameters
    ----------
    cell_totals : output of compute_cell_totals
    fixed_ids   : [4159, 4556] from the assignment spec
    """
    if cell_totals.empty:
        raise ValueError("cell_totals is empty — run compute_cell_totals first.")

    top_square = int(
        cell_totals.nlargest(1, "total_traffic")["square_id"].iloc[0]
    )

    rows = [{"label": "highest_traffic", "square_id": top_square}]
    for sid in fixed_ids:
        rows.append({"label": f"fixed_{sid}", "square_id": int(sid)})

    targets = pd.DataFrame(rows).drop_duplicates("square_id", keep="first")

    known = set(cell_totals["square_id"].astype(int))
    missing = [s for s in targets["square_id"] if s not in known]
    if missing:
        raise ValueError(f"Target squares not found in data: {missing}")

    return targets


# ---------------------------------------------------------------------------
# Target time series (3 areas only)
# ---------------------------------------------------------------------------
def build_area_timeseries(
    files: list[Path],
    square_ids: list[int],
    chunksize: int,
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a continuous time-indexed series for each target square.

    Missing intervals are filled with 0 (no CDR activity).

    Returns
    -------
    timeseries_df : long-format DataFrame [square_id, datetime, internet_traffic]
    gap_summary   : DataFrame [square_id, n_filled_gaps]
    """
    target_set = set(map(int, square_ids))
    parts: list[pd.DataFrame] = []

    for fp in files:
        for chunk, _ in iter_file_chunks(fp, chunksize):
            if chunk.empty:
                continue
            subset = chunk[chunk["square_id"].astype(int).isin(target_set)]
            if not subset.empty:
                parts.append(subset)

    if not parts:
        raise ValueError(f"No data found for square_ids={square_ids}")

    combined = pd.concat(parts, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["time_interval"], unit="ms")

    full_idx = pd.date_range(
        combined["datetime"].min(),
        combined["datetime"].max(),
        freq=freq,
    )

    series_frames: list[pd.DataFrame] = []
    gap_rows: list[dict] = []

    for sid in square_ids:
        sq_data = (
            combined[combined["square_id"].astype(int) == int(sid)]
            .groupby("datetime")["internet_traffic"]
            .sum()
            .sort_index()
        )
        n_gaps = int(len(full_idx.difference(sq_data.index)))
        filled = sq_data.reindex(full_idx, fill_value=0.0)

        series_frames.append(pd.DataFrame({
            "square_id":        np.uint16(sid),
            "datetime":         full_idx,
            "internet_traffic": filled.to_numpy(dtype="float32"),
        }))
        gap_rows.append({"square_id": int(sid), "n_filled_gaps": n_gaps})

    return pd.concat(series_frames, ignore_index=True), pd.DataFrame(gap_rows)


# ---------------------------------------------------------------------------
# City heatmap
# ---------------------------------------------------------------------------
def build_heatmap_grid(cell_totals: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """
    Merge cell totals onto the full grid and compute row/col coordinates.

    Uses numpy reshape to derive row/col instead of element-wise formula,
    which is faster for large grids.
    """
    n = grid_size * grid_size
    all_ids = np.arange(1, n + 1, dtype=np.uint16)
    # Row-major layout: square_id 1 is row 0, col 0
    rows = ((all_ids.astype(int) - 1) // grid_size).astype(np.uint8)
    cols = ((all_ids.astype(int) - 1) % grid_size).astype(np.uint8)

    grid = pd.DataFrame({
        "square_id": all_ids,
        "row":        rows,
        "col":        cols,
    })
    grid = grid.merge(
        cell_totals[["square_id", "total_traffic"]],
        on="square_id",
        how="left",
    )
    grid["is_missing"] = grid["total_traffic"].isna()
    grid["total_traffic"] = grid["total_traffic"].fillna(0.0)
    return grid[["square_id", "row", "col", "total_traffic", "is_missing"]]


# ---------------------------------------------------------------------------
# Memory utility
# ---------------------------------------------------------------------------
def memory_profile(df: pd.DataFrame, label: str) -> dict:
    """Return a memory snapshot dict suitable for building comparison tables."""
    nb = int(df.memory_usage(deep=True).sum())
    return {
        "label":     label,
        "rows":      len(df),
        "columns":   len(df.columns),
        "memory_mb": round(nb / 1024 ** 2, 3),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_outputs(
    cell_totals: pd.DataFrame,
    target_areas: pd.DataFrame,
    timeseries: pd.DataFrame,
    heatmap: pd.DataFrame,
    quality_log: pd.DataFrame,
    processed_dir: Path,
    tables_dir: Path,
) -> None:
    """Write all processed artefacts to disk."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cell_totals.to_csv(processed_dir / "cell_totals.csv", index=False)
    target_areas.to_csv(processed_dir / "target_areas.csv", index=False)
    heatmap.to_csv(processed_dir / "city_heatmap.csv", index=False)
    quality_log.to_csv(tables_dir / "parse_quality_log.csv", index=False)

    try:
        timeseries.to_parquet(processed_dir / "area_timeseries.parquet", index=False)
    except ImportError:
        logger.warning("pyarrow not available — saving timeseries as CSV instead.")
        timeseries.to_csv(processed_dir / "area_timeseries.csv", index=False)

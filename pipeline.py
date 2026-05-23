"""
Command-line data pipeline for the Telecom Italia Milan traffic dataset.

Processes all raw .txt files in data/ and writes the following to processed/:
  - cell_totals.csv         total 2-month traffic per grid square (all 10,000)
  - target_areas.csv        the three focal areas for Tasks 2 and 3
  - area_timeseries.parquet continuous 10-min series for the three target areas
  - city_heatmap.csv        grid-structured totals for the spatial heatmap

Also writes tables/parse_quality_log.csv with per-file data quality counts.

Usage
-----
    python pipeline.py            # uses defaults from src/config.py
    python pipeline.py --force    # reprocess even if outputs already exist
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config
import preprocessing as pp

logging.basicConfig(        
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run(force: bool = False) -> None:
    cell_totals_path  = config.PROCESSED_DIR / "cell_totals.csv"
    timeseries_path   = config.PROCESSED_DIR / "area_timeseries.parquet"

    if not force and cell_totals_path.exists() and timeseries_path.exists():
        log.info("Outputs already exist. Use --force to reprocess.")
        return

    # ------------------------------------------------------------------
    # 1. Discover files
    # ------------------------------------------------------------------
    files = pp.find_raw_files(config.DATA_DIR)
    if not files:
        log.error("No data files found in %s", config.DATA_DIR)
        log.error("Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV")
        sys.exit(1)

    log.info("Found %d data file(s)", len(files))
    if len(files) < 52:
        log.warning(
            "Only %d files found. Task 3 requires ~52 files "
            "(Nov 1 through Dec 22). Proceeding anyway.", len(files)
        )

    # ------------------------------------------------------------------
    # 2. Compute cell totals across all files
    # ------------------------------------------------------------------
    log.info("Computing cell totals across all files...")
    t0 = time.time()
    cell_totals, quality_log = pp.compute_cell_totals(files, config.CHUNKSIZE)
    log.info(
        "Cell totals done: %d squares in %.1fs", len(cell_totals), time.time() - t0
    )

    # ------------------------------------------------------------------
    # 3. Identify the three target areas
    # ------------------------------------------------------------------
    target_areas = pp.identify_target_areas(cell_totals, config.FIXED_SQUARE_IDS)
    log.info("Target areas identified:")
    for _, row in target_areas.iterrows():
        total = cell_totals.loc[
            cell_totals["square_id"] == row["square_id"], "total_traffic"
        ].iloc[0]
        log.info("  %-20s  square_id=%-5d  total=%.2f", row["label"], row["square_id"], total)

    # ------------------------------------------------------------------
    # 4. Build time series for target areas
    # ------------------------------------------------------------------
    target_ids = target_areas["square_id"].tolist()
    log.info("Building time series for squares %s ...", target_ids)
    t0 = time.time()
    timeseries, gap_summary = pp.build_area_timeseries(
        files, target_ids, config.CHUNKSIZE, config.DATA_FREQUENCY
    )
    log.info("Time series done: %d rows in %.1fs", len(timeseries), time.time() - t0)
    for _, row in gap_summary.iterrows():
        log.info("  square_id=%-5d  filled_gaps=%d", row["square_id"], row["n_filled_gaps"])

    # ------------------------------------------------------------------
    # 5. Build heatmap grid
    # ------------------------------------------------------------------
    heatmap = pp.build_heatmap_grid(cell_totals, config.GRID_SIZE)

    # ------------------------------------------------------------------
    # 6. Save all outputs
    # ------------------------------------------------------------------
    pp.save_outputs(
        cell_totals, target_areas, timeseries, heatmap, quality_log,
        config.PROCESSED_DIR, config.TABLES_DIR,
    )
    log.info("All outputs saved to %s", config.PROCESSED_DIR)

    # Summary
    print()
    print("=" * 50)
    print("Pipeline complete")
    print("=" * 50)
    print(f"  Grid squares processed : {len(cell_totals):,}")
    print(f"  Target area rows       : {len(timeseries):,}")
    print(f"  Data quality issues    : {quality_log['total_issues'].sum() if 'total_issues' in quality_log.columns else 'see tables/parse_quality_log.csv'}")
    print(f"  Outputs in            : {config.PROCESSED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess even if output files already exist."
    )
    args = parser.parse_args()
    run(force=args.force)

#!/usr/bin/env python3
"""
Download CRSP daily data for all S&P 500 constituents (historical membership)
and save as a local CSV file for backtesting.

Usage:
    python pipelines/download_sp500_crsp_data.py --years 10

Arguments:
    --years : number of years of history to download (default = 20)

Example:
    python pipelines/download_sp500_crsp_data.py --years 5
"""

import wrds
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time
import argparse

# ---------------------------------------------------------------------
# Parse user input
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--years", type=int, default=20, help="Number of years of data to download")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
END = date.today()
START = (END - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
END_STR = END.strftime("%Y-%m-%d")

OUT_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / f"crsp_sp500_{args.years}yr.csv"
BATCH_SIZE = 100  # number of permnos per SQL query

print(f"Downloading {args.years} years of S&P 500 data ({START} → {END_STR})")
print(f"Output: {OUT_PATH}")

# ---------------------------------------------------------------------
# Connect to WRDS
# ---------------------------------------------------------------------
print("Connecting to WRDS...")
db = wrds.Connection()
print("Connected successfully.")

# ---------------------------------------------------------------------
# Get all PERMNOs that were ever part of the S&P 500
# ---------------------------------------------------------------------
print("Fetching S&P 500 constituents from crsp.dsp500list...")
sp500 = db.raw_sql(f"""
    select distinct permno
    from crsp.dsp500list
    where ending >= '{START}'::date or ending is null
""")
permnos = sp500['permno'].dropna().astype(int).unique().tolist()
print(f"Found {len(permnos)} unique S&P 500 permnos.")

# ---------------------------------------------------------------------
# Function to query in batches
# ---------------------------------------------------------------------
def fetch_batch(batch_permnos):
    permnos_str = ", ".join(str(p) for p in batch_permnos)
    sql = f"""
        select a.permno, a.date, a.prc, a.shrout, a.vol, a.ret,
               coalesce(b.dlret, 0.0) as dlret,
               ((1.0 + a.ret) * (1.0 + coalesce(b.dlret, 0.0)) - 1.0) as ret_total
        from crsp.dsf as a
        left join crsp.dsedelist as b
          on a.permno = b.permno
         and a.date   = b.dlstdt
        where a.date between '{START}' and '{END_STR}'
          and a.permno in ({permnos_str})
        order by a.permno, a.date
    """
    return db.raw_sql(sql, date_cols=['date'])

# ---------------------------------------------------------------------
# Download data in manageable chunks
# ---------------------------------------------------------------------
all_data = []
for i in range(0, len(permnos), BATCH_SIZE):
    batch_permnos = permnos[i:i + BATCH_SIZE]
    print(f"Fetching batch {i // BATCH_SIZE + 1} "
          f"({i + 1}-{i + len(batch_permnos)} of {len(permnos)})...")
    batch_df = fetch_batch(batch_permnos)
    all_data.append(batch_df)
    time.sleep(1)  # delay to reduce WRDS load

db.close()

# Combine all batches
df = pd.concat(all_data, ignore_index=True)
print(f"Total rows downloaded: {len(df):,}")

# ---------------------------------------------------------------------
# Save as CSV
# ---------------------------------------------------------------------
print(f"Saving to {OUT_PATH} ...")
df.to_csv(OUT_PATH, index=False)
print("Done. CRSP S&P 500 data successfully saved as CSV.")

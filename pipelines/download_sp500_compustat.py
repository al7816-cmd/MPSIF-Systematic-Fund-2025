#!/usr/bin/env python3
"""
Download Compustat daily data for all S&P 500 constituents whose membership
overlaps the last N years, compute proper daily total returns including dividends,
and save as PARQUET.

Return formula implemented:
    adj_prc      = prccd / ajexdi
    trfd_ffill   = forward-fill trfd within each permno, fill missing with 1.0
    adj_tr_price = adj_prc * trfd_ffill
    ret_daily    = adj_tr_price / adj_tr_price.shift(1) - 1

This provides CRSP-like total returns including dividends.

Usage:
    python pipelines/download_sp500_compustat_data.py --years 20
"""

import wrds
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time
import argparse


# ------------------------------------------------------------
# Parse input
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--years", type=int, default=20,
                    help="Number of years of data to download")
args = parser.parse_args()

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
END = date.today()
START = (END - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
END_STR = END.strftime("%Y-%m-%d")

OUT_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / f"compustat_sp500_{args.years}yr.parquet"

BATCH_SIZE = 100

print(f"Downloading {args.years} years of S&P 500 Compustat daily data")
print(f"Window: {START} → {END_STR}")
print(f"Saving to: {OUT_PATH}")


# ------------------------------------------------------------
# Connect to WRDS
# ------------------------------------------------------------
print("Connecting to WRDS...")
db = wrds.Connection()
print("Connected.")


# ------------------------------------------------------------
# Get S&P 500 PERMNOs whose membership intersects window
# ------------------------------------------------------------
print("Fetching S&P 500 constituent PERMNOs...")

sp500 = db.raw_sql(f"""
    SELECT DISTINCT permno
    FROM crsp.dsp500list
    WHERE start <= '{END_STR}'::date
      AND (ending >= '{START}'::date OR ending IS NULL)
""")

permnos = sp500["permno"].dropna().astype(int).unique().tolist()
print(f"Found {len(permnos)} valid S&P 500 PERMNOs.")

if not permnos:
    raise RuntimeError("No S&P 500 constituents in date window.")


# ------------------------------------------------------------
# Batch fetcher
# ------------------------------------------------------------
def fetch_batch(batch_permnos):

    permnos_str = ", ".join(str(p) for p in batch_permnos)

    sql = f"""
        WITH linked AS (
            SELECT
                l.lpermno AS permno,
                l.gvkey,
                l.liid AS iid,
                l.linkdt,
                COALESCE(l.linkenddt, DATE '9999-12-31') AS linkenddt
            FROM crsp.ccmxpf_linktable AS l
            WHERE l.linktype IN ('LU', 'LC')
              AND l.linkprim IN ('P', 'C')
              AND l.lpermno IN ({permnos_str})
        ),

        sec AS (
            SELECT
                gvkey,
                iid,
                datadate,
                prccd,
                ajexdi,
                trfd
            FROM comp.secd
            WHERE datadate BETWEEN '{START}'::date AND '{END_STR}'::date
        ),

        joined AS (
            SELECT
                lk.permno,
                sec.gvkey,
                sec.iid,
                sec.datadate AS date,
                sec.prccd,
                sec.ajexdi,
                sec.trfd,
                (sec.prccd / NULLIF(sec.ajexdi, 0)) AS adj_prc
            FROM sec
            JOIN linked AS lk
              ON sec.gvkey = lk.gvkey
             AND sec.iid = lk.iid
             AND sec.datadate BETWEEN lk.linkdt AND lk.linkenddt
        )

        SELECT
            permno,
            date,
            prccd,
            ajexdi,
            trfd,
            adj_prc
        FROM joined
        ORDER BY permno, date
    """

    return db.raw_sql(sql, date_cols=["date"])


# ------------------------------------------------------------
# Download in batches
# ------------------------------------------------------------
all_data = []

for i in range(0, len(permnos), BATCH_SIZE):
    batch = permnos[i:i+BATCH_SIZE]
    print(f"Batch {i//BATCH_SIZE+1}: PERMNOs {i+1}–{i+len(batch)} of {len(permnos)}")
    df_batch = fetch_batch(batch)
    all_data.append(df_batch)
    time.sleep(1)

db.close()

df = pd.concat(all_data, ignore_index=True)
print(f"Total rows downloaded: {len(df):,}")


# ------------------------------------------------------------
# Compute proper daily total returns
# ------------------------------------------------------------
print("Computing daily adjusted prices and returns...")

df = df.sort_values(["permno", "date"])

# 1. Split-adjusted price already computed as adj_prc

# 2. Forward-fill dividend factor trfd
df["trfd_ffill"] = df.groupby("permno")["trfd"].ffill().fillna(1.0)

# 3. Total-return adjusted price
df["adj_tr_price"] = df["adj_prc"] * df["trfd_ffill"]

# 4. Daily total return (CRSP-like)
df["ret_daily"] = df.groupby("permno")["adj_tr_price"].pct_change()


# ------------------------------------------------------------
# Save to parquet
# ------------------------------------------------------------
print(f"Saving Parquet → {OUT_PATH}")
df.to_parquet(OUT_PATH, index=False)
print("Done. File written successfully.")

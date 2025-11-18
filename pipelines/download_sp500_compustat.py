#!/usr/bin/env python3
"""
Download Compustat daily data for all S&P 500 constituents whose membership
overlaps the last N years, and save as PARQUET.

This version:
  - Computes correct total-return daily returns (`ret`)
  - Uses adj_prc_tr = (price/ajexdi) * trfd for dividend- and split-adjusted prices
  - Ensures LAG is partitioned by permno and ordered by date
"""

import wrds
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time
import argparse

# ---------------------------------------------------------------------
# Parse input
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--years", type=int, default=20,
                    help="Number of years to download")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------
END = date.today()
START = (END - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
END_STR = END.strftime("%Y-%m-%d")

OUT_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / f"compustat_sp500_{args.years}yr.parquet"

BATCH_SIZE = 100

print(f"Downloading S&P500 {args.years}yr data: {START} → {END_STR}")
print(f"Output: {OUT_PATH}")

# ---------------------------------------------------------------------
# WRDS connect
# ---------------------------------------------------------------------
print("Connecting to WRDS...")
db = wrds.Connection()
print("Connected.")

# ---------------------------------------------------------------------
# S&P 500 membership (correct intersection logic)
# ---------------------------------------------------------------------
print("Fetching S&P 500 permnos...")

sp500 = db.raw_sql(f"""
    SELECT DISTINCT permno
    FROM crsp.dsp500list
    WHERE start <= '{END_STR}'::date
      AND (ending >= '{START}'::date OR ending IS NULL)
""")

permnos = sp500["permno"].dropna().astype(int).unique().tolist()
print(f"Found {len(permnos)} permnos.")

if not permnos:
    raise RuntimeError("No S&P 500 constituents found.")

# ---------------------------------------------------------------------
# Batch fetch
# ---------------------------------------------------------------------
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
            FROM crsp.ccmxpf_linktable l
            WHERE l.linktype IN ('LU','LC')
              AND l.linkprim IN ('P','C')
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
                (sec.prccd / NULLIF(sec.ajexdi, 0)) AS adj_prc,
                (sec.prccd / NULLIF(sec.ajexdi, 0)) * sec.trfd AS adj_prc_tr
            FROM sec
            JOIN linked lk
              ON sec.gvkey = lk.gvkey
             AND sec.iid   = lk.iid
             AND sec.datadate BETWEEN lk.linkdt AND lk.linkenddt
        )

        SELECT
            permno,
            date,
            prccd,
            ajexdi,
            trfd,
            adj_prc,
            adj_prc_tr,
            CASE
                WHEN LAG(adj_prc_tr) OVER (
                        PARTITION BY permno
                        ORDER BY date
                ) IS NULL
                THEN NULL
                ELSE adj_prc_tr /
                     LAG(adj_prc_tr) OVER (
                        PARTITION BY permno
                        ORDER BY date
                     ) - 1.0
            END AS ret
        FROM joined
        ORDER BY permno, date
    """

    return db.raw_sql(sql, date_cols=["date"])


# ---------------------------------------------------------------------
# Download loop
# ---------------------------------------------------------------------
all_data = []

for i in range(0, len(permnos), BATCH_SIZE):
    batch = permnos[i : i + BATCH_SIZE]
    print(f"Batch {i//BATCH_SIZE+1}: permnos {i+1}-{i+len(batch)}")
    df_batch = fetch_batch(batch)
    all_data.append(df_batch)
    time.sleep(1)

db.close()

df = pd.concat(all_data, ignore_index=True)
print(f"Total rows: {len(df):,}")

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
print(f"Saving parquet → {OUT_PATH}")
df.to_parquet(OUT_PATH, index=False)
print("Done.")

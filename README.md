## Quick Start: Connect to WRDS Cloud on HPC

### 1. Load Python and set up environment
module load python/3.10        # or similar on your cluster
python3 -m venv ~/wrds_venv
source ~/wrds_venv/bin/activate
pip install wrds pandas psycopg2-binary sqlalchemy

### 2. (Optional) Create WRDS password file for non-interactive jobs
echo "wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_WRDS_USERNAME:YOUR_WRDS_PASSWORD" > ~/.pgpass
chmod 600 ~/.pgpass

### 3. Test connection
python3 - <<'PYCODE'
import wrds
db = wrds.Connection()          # uses ~/.pgpass if present
print(db.list_libraries()[:5])  # should show libraries like ['crsp', 'comp', 'ff', ...]
db.close()
PYCODE

### 4. Example query: CRSP daily prices
python3 - <<'PYCODE'
import pandas as pd, wrds
db = wrds.Connection()
sql = """
select permno, date, prc, shrout, ret
from crsp.dsf
where date between '2024-01-01' and '2024-03-31'
limit 10;
"""
df = db.raw_sql(sql, date_cols=['date'])
print(df.head())
db.close()
PYCODE

If you see output rows, your WRDS connection is working correctly.

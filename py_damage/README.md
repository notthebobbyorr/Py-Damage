# Damage Pipeline (Python)

## Pipeline

Run the polars pipeline to pull Statcast data (MLB only by default):

```
python pipeline.py --start-date 2025-03-01 --end-date 2025-10-01 --season 2025 --output-dir .. --output-tag 2021_2024
```

Notes:
- Use `--input-csv` if you already have Statcast CSVs.
- Use `--positions-csv` to join hitter position counts (see `damage_pos_2021_2024.csv` columns).
- The script writes CSVs expected by the Streamlit app (ex: `new_team_damage.csv`).

## Streamlit app

From `R_Damage`:

```
streamlit run damage_streamlit.py
```

# Official hitter totals pilot

Individual Hitter Stats, Percentiles, Hitter Comps, and Auto Regressed expand rows for regular-season
totals. Data is keyed by MLB player ID, season, and level, across all teams. No API
requests occur on page load or click. MLB, Triple-A, Low-A, and Low Minors are supported. Unmatched rows show an unavailable message rather than another level's stats. Other game
types retain the standard table. In Comps, postseason rows show unavailable details; MLB-equivalent targets retain their original level for official totals.

`data/output/official_hitting.parquet` contains the 2015–2026 backfill, retrieval
timestamps, source counts, official slash-line rates, OPS, and BABIP. K% and BB% use PA.
Display order: GP, PA, AB, H, HR, AVG/OBP/SLG, OPS, BABIP, K%, BB%, R, RBI, SB, CS.
Missing rates display a dash; zeros remain zeros.

## Refresh

Run with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe pipeline/build_official_hitting.py --seasons 2026
```

The script imports `baseball_data` from the sibling `codex_tmp/scraper` directory.
Use `--scraper-path` when that package lives elsewhere. This package is needed
only on the machine running the refresh, not on Streamlit Cloud. No installation
is required. `run_daily_refresh.py` runs the current-season update for all four levels before
recording completion. Historical seasons can be explicitly refreshed with
`--seasons 2015 2016 ...`. A failed retrieval leaves the previous file untouched.
After refreshing in an already-running dev process, clear Streamlit's resource
cache or restart the app to load the new snapshot.

Use `--levels 1 11 14 16` to select sport IDs explicitly. Seasons without official stats (for example cancelled minor-league seasons) are reported as zero rows.

## Interaction and rollback

Click a row, or focus it and press Enter/Space, to toggle its detail row. Several
rows may be open. Column-header sorting sorts the displayed page and moves
details with their hitter. The existing Sort by controls sort the entire filtered
result before pagination. Changing page or filters resets expanded rows.

The pilot uses a custom HTML table with the existing conditional colors. Existing
filters, Create-a-Plot, sort controls, pagination, and the CSV button remain.
The native dataframe toolbar (column resizing/hiding, search, fullscreen) is not
part of this pilot. The official detail row is not added to the modeled-stat CSV.

To roll back the UI, remove the `official_hitting_details` argument from the main
hitter page's `render_table` call; the standard renderer remains the default.

Player sabermetrics: official wRC+ follows OPS for hitters; xFIP follows ERA for pitchers. These are fetched during daily refresh with complete pagination and player-season-level matching. Missing API values remain blank (displayed as a dash), and team tables omit these metrics.

Fullscreen viewing is available using the table’s Fullscreen button. Exit fullscreen or Escape restores the table in place, preserving expanded rows and column sorting.

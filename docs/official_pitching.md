# Official pitching popouts

Enabled on Pitcher Individual Stats, both Percentiles modes, Comps (target and results), and Auto Regressed. Regular-season only, at MLB, Triple-A, Low-A, and Low Minors levels. Comps use original player/season/level identifiers, not the translated level used for coloring.

Display order: G GS IP W-L QS ERA WHIP K% BB% K-BB% AVG BABIP OPS TBF SO HR HA SV BS Hld Inh. Runners %.

Hld means holds; HA means hits allowed; Inh. Runners % = 100 * inherited runners scored / inherited runners (missing when none inherited). QS and BABIP come from fully paginated advanced totals matched to the same season and level. Missing source values are not assumed to be zero. OPS and AVG are opponent rates.

K% = 100 * SO / TBF; BB% = 100 * BB / TBF; K-BB% is their difference. A zero denominator displays missing rates. IP remains the official baseball string (.1 or .2 are outs, not decimal innings). Undefined official rates remain missing.

Refresh with `.venv/Scripts/python.exe pipeline/build_official_pitching.py --seasons 2026`. Defaults to sport IDs 1, 11, 14, 16. Use --levels and --scraper-path as needed; the scraper defaults to sibling codex_tmp/scraper. The daily refresh invokes this before reporting completion. Fetch failures preserve the prior Parquet snapshot. Restart or clear resource caches to reload an updated snapshot in a running app. Runtime needs only the saved data, no scraper or network.

To roll back an individual page, remove its official_pitching_details argument. The shared custom table retains the same capabilities and toolbar limitations documented in official_hitting_pilot.md.

## Additional fields observed in StatsAPI (2026 sample)

Standard: holds, blown saves, save opportunities, games finished, complete games, shutouts, earned runs, runs allowed, walks, intentional walks, hit batters, wild pitches, balks, pickoffs, inherited runners/scored, stolen bases/caught stealing, doubles/triples allowed, opponent OBP/SLG/OPS, pitch count, strikes/strike percentage, pitches per inning, K/9, BB/9, H/9, HR/9, K/BB, ground outs/air outs.

Advanced: quality starts, BABIP, bequeathed runners/scored, swing-and-miss counts, whiff percentage, fly-ball percentage, run support, pitches per PA, and ground/fly/line/pop hit/out counts.

Sabermetrics: FIP, xFIP, FIP-, ERA-, WAR, RA9-WAR, RAR, leverage indices (pLI, inLI, gmLI, exLI). These are fields returned by MLB's endpoint, not a claim that all are official scoring statistics or available for every season/level.

LOB% was not present in the standard, advanced, or sabermetric responses inspected. It has not been calculated or added. An estimated strand rate would need separate labeling and agreement on its definition.

Player sabermetrics: official wRC+ follows OPS for hitters; xFIP follows ERA for pitchers. These are fetched during daily refresh with complete pagination and player-season-level matching. Missing API values remain blank (displayed as a dash), and team tables omit these metrics.

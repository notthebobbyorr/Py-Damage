# Official team popouts

Team Hitting and Team Pitching Season Stats tables expand on click in Regular Season mode. The existing player columns are reused, with R and R/G between HR and AVG/OBP/SLG, and RA9 after ERA. Team pitching omits inherited runners. Hld means holds. Missing source fields display a dash.

R/G = runs / team games played. RA9 = 27 * runs allowed / outs pitched; baseball innings .1 and .2 represent one and two outs. Zero denominators display a dash.

The saved data covers the supported seasons and levels. The current team model tables only contain MLB rows; other levels can display official details when model rows become available. Team code, season, and level must match uniquely. Ambiguous historical minor-league abbreviations remain unavailable rather than attaching a different club’s totals. Historical OAK/ATH and AZ/ARI aliases are normalized. No totals are summed from player rows.

Refresh with `.venv/Scripts/python.exe pipeline/build_official_teams.py --seasons 2026`. This runs during the daily refresh using the existing local scraper transport. Standard and advanced team endpoints are paginated and saved atomically to data/output/official_teams.parquet. The app only reads that file via cache_resource.

The shared HTML component supports click/keyboard expand-collapse and column sorting; it has the same native-dataframe toolbar limitations as the player popouts. Remove the team detail flags from the two Season Stats render_table calls to roll back the UI.

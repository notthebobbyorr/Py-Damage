"""Add missing pooled swing counts to team/league tables without changing metrics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.backfill_miss_distance import aggregate_for, scope_for
from pipeline.backfill_bat_tracking import OUT, write_updated
from pipeline.data_aggregate import _add_game_type_group, _tag_pitch
from pipeline.swing_rates import swing_count_expressions

COUNTS = ['rate_whiffs', 'rate_swings', 'rate_pitches']
STEMS = {'new_team_damage', 'team_hitter_splits', 'new_hitting_lg_avg',
         'new_team_stuff', 'team_pitcher_splits', 'new_lg_stuff', 'league_pitch_types'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    backup_root = ROOT / 'data/backups/swing_counts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    paths = [p for p in OUT.rglob('*.parquet') if p.stem.split('.')[0] in STEMS]
    columns = ['batter_mlbid', 'pitcher_mlbid', 'season', 'level_id', 'game_type', 'game_pk',
        'game_date', 'hitting_code', 'pitching_code', 'home_team', 'away_team', 'half_inning',
        'home_mlbid', 'away_mlbid', 'throws', 'stands', 'pitcher_hand', 'pi_pitch_group',
        'pi_pitch_type', 'pi_pitch_sub_type', 'park_mlbid', 'whiff', 'swing']
    for source in args.input:
        scan = pl.scan_parquet(source)
        if source.name == 'pitch_data_2015_2025_all_levels.parquet':
            scan = scan.filter(pl.col('season') < 2023)
        for year in sorted(scan.select('season').unique().collect()['season'].to_list()):
            pitch = _tag_pitch(_add_game_type_group(scan.filter(pl.col('season') == year).select(columns).collect()))
            cache = {}
            for path in paths:
                if '.season_' in path.name and f'.season_{year}.' not in path.name: continue
                frame = pl.read_parquet(path)
                selected = frame.filter(pl.col('season') == year)
                if selected.is_empty(): continue
                kind, mode = scope_for(path)
                if (kind, mode) not in cache:
                    cache[(kind, mode)] = aggregate_for(pitch, kind, mode, frame, expressions=swing_count_expressions)
                counts, keys = cache[(kind, mode)]
                for key in keys:
                    counts = counts.filter(pl.col(key).is_in(selected[key].cast(counts.schema[key]).unique().implode()))
                counts = counts.cast({k: selected.schema[k] for k in keys})
                updated = selected.drop(COUNTS, strict=False).join(counts, on=keys, how='left', validate='m:1', maintain_order='left')
                old_columns = [c for c in selected.columns if c not in COUNTS]
                if not updated.select(old_columns).equals(selected.select(old_columns)):
                    raise ValueError('Swing counts changed unrelated statistics')
                positions = frame.with_row_index('__position').filter(pl.col('season') == year)['__position']
                updated = updated.with_columns(positions)
                rest = frame.with_row_index('__position').filter(pl.col('season') != year)
                result = pl.concat([rest, updated], how='diagonal_relaxed').sort('__position').drop('__position')
                write_updated(result, path, backup_root)
            print(f'Added swing counts: {year}', flush=True)
    files = sorted(str(p.relative_to(backup_root/'output')) for p in (backup_root/'output').rglob('*.parquet'))
    (backup_root/'report.json').write_text(json.dumps({'updated_tables':files}, indent=2))
    print(f'Swing-count backups: {backup_root}', flush=True)


if __name__ == '__main__':
    main()

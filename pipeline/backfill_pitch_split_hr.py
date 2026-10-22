"""Add regular-season HR counts to saved hitter/team/league pitch-group splits."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import json
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.backfill_bat_tracking import OUT, write_updated
from pipeline.backfill_miss_distance import aggregate_for, scope_for
from pipeline.data_aggregate import _add_game_type_group, _tag_pitch


def hr_expressions(frame):
    return [(pl.col('pitch_outcome') == 'HR').sum().alias('HR')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    backup_root = ROOT / 'data/backups/pitch_split_hr' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    paths = [p for p in OUT.rglob('*.parquet') if (scope := scope_for(p))
             and scope[0] in {'hitter', 'team_hitter', 'league_hitter'} and scope[1] in {'group', 'combined'}]
    columns = ['batter_mlbid', 'season', 'level_id', 'game_type', 'game_pk', 'game_date',
        'hitting_code', 'pitching_code', 'home_team', 'away_team', 'half_inning', 'home_mlbid',
        'away_mlbid', 'throws', 'stands', 'pitcher_hand', 'pi_pitch_group', 'pi_pitch_type',
        'pi_pitch_sub_type', 'pitch_outcome']
    for source in args.input:
        scan = pl.scan_parquet(source)
        if source.name == 'pitch_data_2015_2025_all_levels.parquet':
            scan = scan.filter(pl.col('season') < 2023)
        for year in sorted(scan.select('season').unique().collect()['season'].to_list()):
            pitch = _tag_pitch(_add_game_type_group(scan.filter(
                (pl.col('season') == year) & (pl.col('pitch_outcome') == 'HR')).select(columns).collect()))
            cache = {}
            for path in paths:
                if path.name.startswith('season_') and not path.name.startswith(f'season_{year}.'):
                    continue
                frame = pl.read_parquet(path)
                selected = frame.filter(pl.col('season') == year)
                if selected.is_empty():
                    continue
                kind, mode = scope_for(path)
                dims = path.parent.name.split('__') if mode == 'combined' else None
                key = (kind, mode, tuple(dims or []))
                if key not in cache:
                    cache[key] = aggregate_for(pitch, kind, mode, frame, dims, expressions=hr_expressions)
                rates, keys = cache[key]
                counts = rates.cast({k: selected.schema[k] for k in keys})
                updated = selected.drop('HR', strict=False).join(counts, on=keys, how='left',
                    validate='m:1', maintain_order='left').with_columns(pl.col('HR').fill_null(0))
                original = [c for c in selected.columns if c != 'HR']
                assert updated.select(original).equals(selected.select(original)), path
                positions = frame.with_row_index('__position').filter(pl.col('season') == year)['__position']
                updated = updated.with_columns(positions)
                rest = frame.with_row_index('__position').filter(pl.col('season') != year)
                result = pl.concat([rest, updated], how='diagonal_relaxed').sort('__position').drop('__position')
                write_updated(result, path, backup_root)
            print(f'Added pitch-group HR counts: {year}', flush=True)
    files = sorted(str(p.relative_to(backup_root / 'output')) for p in (backup_root / 'output').rglob('*.parquet'))
    (backup_root / 'report.json').write_text(json.dumps({'updated_tables': files}, indent=2))
    print(f'Updated {len(files)} files; backups: {backup_root}', flush=True)


if __name__ == '__main__':
    main()

"""Refresh observed miss distance without recalculating other app statistics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.backfill_bat_tracking import write_updated, OUT
from pipeline.build_combined_splits import add_dimensions
from pipeline.data_aggregate import (
    _add_game_type_group, _tag_pitch, _normalize_team_codes, _split_vs_lr,
    _split_home_away, _split_month, _split_half, _with_split,
)
from pipeline.pitch_split_games import apply_game_classifications, CACHE
from pipeline.tracking_metrics import miss_distance_expressions

METRICS = ['miss_distance', 'miss_distance_n']


def scope_for(path):
    if 'combined_splits' in path.parts:
        return path.parts[-3], 'combined'
    stem = path.stem.split('.')[0]
    if stem.startswith('damage_pos_'): return 'hitter', 'season'
    for kind, stems in {
        'hitter': ['hitter_pctiles', 'hitters_regressed', 'hitter_splits', 'hitter_gamelogs', 'hitter_pitch_splits'],
        'pitcher': ['pitcher_stuff_new', 'pitcher_pctiles', 'pitchers_regressed', 'pitcher_splits', 'pitcher_gamelogs'],
        'pitch': ['new_pitch_types', 'pitch_types_pctiles', 'pitch_types_regressed', 'pitch_types_splits', 'pitch_type_gamelogs'],
        'team_hitter': ['new_team_damage', 'team_hitter_splits', 'team_hitting_pitch_splits'],
        'team_pitcher': ['new_team_stuff', 'team_pitcher_splits'],
        'league_hitter': ['new_hitting_lg_avg', 'league_hitting_pitch_splits'],
        'league_pitcher': ['new_lg_stuff'],
        'league_pitch': ['league_pitch_types'],
        'park': ['park_data'],
    }.items():
        if stem in stems:
            mode = ('group' if stem.endswith('pitch_splits') and stem != 'pitch_types_splits'
                    else 'splits' if stem.endswith('_splits') else 'gamelog' if stem.endswith('_gamelogs') else 'season')
            return kind, mode
    return None


def aggregate_for(pitch, kind, mode, frame, dims=None, expressions=miss_distance_expressions):
    hitting = kind in {'hitter', 'team_hitter', 'league_hitter'}
    identity = {'hitter': ['batter_mlbid'], 'pitcher': ['pitcher_mlbid'], 'pitch': ['pitcher_mlbid', 'pitch_tag'],
                'team_hitter': ['hitting_code'], 'team_pitcher': ['pitching_code'],
                'league_hitter': [], 'league_pitcher': [], 'league_pitch': ['pitch_tag', 'pitcher_hand'],
                'park': ['park_mlbid', 'home_team', 'stands']}[kind]
    keys = identity + [c for c in ['season', 'level_id', 'game_type_group'] if c in frame.columns]
    if kind.startswith('team_'):
        pitch = _normalize_team_codes(pitch, identity[0])
    if mode == 'gamelog':
        keys += ['game_pk']
        team_col = 'hitting_code' if hitting else 'pitching_code'
        if team_col in frame.columns: keys.append(team_col)
    if mode in {'combined', 'group'}:
        pitch = apply_game_classifications(pitch.lazy(), pl.read_parquet(CACHE)).filter(pl.col('game_type') == 'R').collect()
        pitch = add_dimensions(pitch, hitting)
        if kind == 'team_hitter':
            pitch = pitch.with_columns(pl.col('hitting_code').replace('OAK', 'ATH'))
        if mode == 'combined':
            keys += dims
            pitch = pitch.drop_nulls(dims)
        else:
            pitch = pitch.with_columns(pl.col('split_pitch_group').replace_strict(
                {'Fastball': 'FA', 'Sinker': 'SI', 'Breaking': 'BR', 'Offspeed': 'OFF'}, default=None).alias('pitch_group'))
            keys += ['pitch_group']
            pitch = pitch.drop_nulls('pitch_group')
    if mode == 'splits':
        parts = []
        for label, part in [
            ('vs L/R', _with_split(pitch, _split_vs_lr(pitch, 'throws' if hitting else 'stands'))),
            ('Home/Away', _with_split(pitch, _split_home_away(pitch, hitting))),
            ('Monthly', _with_split(pitch, _split_month(pitch))),
            ('1st Half/2nd Half', _split_half(pitch)),
        ]:
            parts.append(part.drop_nulls('__split').group_by(keys + ['__split']).agg(
                expressions(part)).rename({'__split': 'split'}).with_columns(pl.lit(label).alias('split_type')))
        return pl.concat(parts, how='diagonal_relaxed'), keys + ['split_type', 'split']
    return pitch.group_by(keys).agg(expressions(pitch)), keys


def merge_metric(frame, metrics, keys):
    for key in keys:
        metrics = metrics.filter(pl.col(key).is_in(frame[key].cast(metrics.schema[key]).unique().implode()))
    metrics = metrics.select(keys + METRICS).cast({k: frame.schema[k] for k in keys})
    result = frame.drop(METRICS, strict=False).join(metrics, on=keys, how='left', validate='m:1', maintain_order='left')
    result = result.with_columns(pl.col('miss_distance_n').fill_null(0))
    original = [c for c in frame.columns if c not in METRICS]
    if len(result) != len(frame) or not result.select(original).equals(frame.select(original)):
        raise ValueError('Miss-distance join changed unrelated statistics')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--backup-root', type=Path, help='Reuse the original backups when resuming an interrupted refresh')
    args = parser.parse_args()
    backup_root = args.backup_root or ROOT / 'data/backups/miss_distance' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    paths = [p for p in OUT.rglob('*.parquet') if scope_for(p)]
    changed = set()
    # Establish a visible blank column even in seasons before tracking existed.
    for path in paths:
        frame = pl.read_parquet(path)
        if not set(METRICS).issubset(frame.columns):
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias('miss_distance'),
                                       pl.lit(0, dtype=pl.UInt32).alias('miss_distance_n'))
            write_updated(frame, path, backup_root)
            changed.add(str(path.relative_to(OUT)))
    columns = ['batter_mlbid', 'pitcher_mlbid', 'season', 'level_id', 'game_type', 'game_pk',
        'game_date', 'hitting_code', 'pitching_code', 'home_team', 'away_team', 'half_inning',
        'home_mlbid', 'away_mlbid', 'throws', 'stands', 'pitcher_hand', 'pi_pitch_group',
        'pi_pitch_type', 'pi_pitch_sub_type', 'park_mlbid', 'miss_distance']
    for path in args.input:
        scan = pl.scan_parquet(path)
        schema = scan.collect_schema()
        for year in sorted(scan.select('season').unique().collect()['season'].to_list()):
            if year < 2023: continue
            # Unmeasured pitches contribute neither distance nor sample weight.
            # Absent groups merge back as blank means and zero recorded samples.
            pitch = _tag_pitch(_add_game_type_group(scan.filter(
                (pl.col('season') == year) & pl.col('miss_distance').is_not_null()).select(columns).collect()))
            cache = {}
            for target in paths:
                if '.season_' in target.name and f'.season_{year}.' not in target.name: continue
                if target.name.startswith('season_') and not target.name.startswith(f'season_{year}.'): continue
                frame = pl.read_parquet(target)
                selected = frame.filter(pl.col('season') == year)
                if selected.is_empty(): continue
                kind, mode = scope_for(target)
                dims = target.parent.name.split('__') if mode == 'combined' else None
                key = (kind, mode, tuple(dims or []), tuple(c for c in ['game_type_group', 'hitting_code', 'pitching_code'] if c in frame.columns))
                if key not in cache:
                    cache[key] = aggregate_for(pitch, kind, mode, frame, dims)
                metrics, keys = cache[key]
                updated = merge_metric(selected, metrics, keys)
                positions = frame.with_row_index('__position').filter(pl.col('season') == year)['__position']
                updated = updated.with_columns(positions)
                rest = frame.with_row_index('__position').filter(pl.col('season') != year)
                result = pl.concat([rest, updated], how='diagonal_relaxed').sort('__position').drop('__position')
                if not result.equals(frame):
                    write_updated(result, target, backup_root)
                    changed.add(str(target.relative_to(OUT)))
            print(f'Updated miss distance: {year}', flush=True)
    backup_root.mkdir(parents=True, exist_ok=True)
    updated = sorted(str(p.relative_to(backup_root / 'output'))
                     for p in (backup_root / 'output').rglob('*.parquet'))
    (backup_root / 'report.json').write_text(json.dumps({'updated_tables': updated}, indent=2), encoding='utf-8')
    print(f'Updated {len(updated)} tables; backups: {backup_root}', flush=True)


if __name__ == '__main__':
    main()

"""Fill missing raw measurements, then refresh only tracking-dependent outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.scrape_bat_tracking import FIELDS, KEYS
from pipeline.data_aggregate import (
    _add_game_type_group, _normalize_team_codes, _split_vs_lr, _split_home_away,
    _split_month, _split_half, _with_split, bat_tracking_expressions,
)
from pipeline.build_combined_splits import add_dimensions
from pipeline.pitch_split_games import apply_game_classifications, CACHE

OUT = ROOT / 'data/output'
BASE_KEYS = ['season', 'level_id', 'game_type_group']
OUTPUT_FIELDS = FIELDS[:5] + ['intercept_x_inches', 'intercept_y_inches', 'fast_swing_pct', 'miss_distance']
TRACKING = OUTPUT_FIELDS + [c + '_n' for c in OUTPUT_FIELDS]
SOURCE_KEYS = ['game_pk', 'at_bat_index', 'pitch_of_ab', 'batter_mlbid', 'pitcher_mlbid']


def fill_measurements(raw, measurements):
    """Keep source rows and existing values; require both matchup identities."""
    if measurements.select(KEYS).is_duplicated().any():
        raise ValueError('Duplicate scraper pitch identities')
    source = measurements.select(
        'game_pk', (pl.col('at_bat_number') - 1).alias('at_bat_index'),
        pl.col('pitch_number').alias('pitch_of_ab'), pl.col('batter').alias('batter_mlbid'),
        pl.col('pitcher').alias('pitcher_mlbid'),
        *[pl.col(c).alias('__scrape_' + c) for c in FIELDS],
    )
    result = raw.join(source.lazy(), on=SOURCE_KEYS, how='left', validate='m:1', maintain_order='left')
    schema = raw.collect_schema()
    return result.with_columns([
        pl.coalesce(pl.col(c).cast(pl.Float64, strict=False) if c in schema else pl.lit(None),
                    pl.col('__scrape_' + c)).alias(c) for c in FIELDS
    ]).drop(['__scrape_' + c for c in FIELDS])


def backup(path, backup_root, relative):
    target = backup_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not target.exists():
        shutil.copy2(path, target)


def enrich_raw(path, destination, measurements, backup_root, min_season=None):
    scan = pl.scan_parquet(path)
    if min_season is not None:
        scan = scan.filter(pl.col('season') >= min_season)
    result = fill_measurements(scan, measurements)
    schema = scan.collect_schema()
    before = scan.select(pl.len().alias('rows'), *[
        (pl.col(c).cast(pl.Float64, strict=False).count() if c in schema else pl.lit(0)).alias(c) for c in FIELDS
    ]).collect().to_dicts()[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.tracking.tmp.parquet')
    result.sink_parquet(temp)
    after = pl.scan_parquet(temp).select(pl.len().alias('rows'), *[
        pl.col(c).count().alias(c) for c in FIELDS
    ]).collect().to_dicts()[0]
    if before['rows'] != after['rows'] or any(after[c] < before[c] for c in FIELDS):
        raise ValueError('Raw backfill lost rows or measurements')
    backup(destination, backup_root, Path('raw') / destination.name)
    temp.replace(destination)
    print(f'Enriched {destination.name}: {before} -> {after}', flush=True)
    return {'source': str(path), 'output': str(destination), 'before': before, 'after': after}


def replace_tracking(frame, metrics, keys):
    """Replace tracking statistics only, preserving row order and other columns."""
    columns = [c for c in TRACKING if c in frame.columns and c in metrics.columns]
    if not columns:
        return frame
    # Restrict identities before narrowing small integer key dtypes.
    for key in keys:
        metrics = metrics.filter(pl.col(key).is_in(frame[key].cast(metrics.schema[key]).unique().implode()))
    metrics = metrics.select(keys + columns).cast({k: frame.schema[k] for k in keys})
    joined = frame.join(metrics.rename({c: '__new_' + c for c in columns}), on=keys,
                        how='left', validate='m:1', maintain_order='left')
    joined = joined.with_columns([pl.col('__new_' + c).alias(c) for c in columns]).drop(
        ['__new_' + c for c in columns])
    if len(joined) != len(frame) or not joined.drop(columns).equals(frame.drop(columns)):
        raise ValueError('Tracking refresh changed unrelated data')
    return joined


def write_updated(frame, path, backup_root):
    path = path.resolve()
    backup(path, backup_root, Path('output') / path.relative_to(OUT))
    from run_daily_refresh import PANDAS_OPTIMIZED_TABLES
    temp = path.with_suffix('.tracking.tmp.parquet')
    if path.stem.split('.')[0] in PANDAS_OPTIMIZED_TABLES:
        # Dictionary strings retain compact app loading without downcasting
        # unrelated Float64 values or converting their NaNs to nulls.
        frame = frame.with_columns(pl.col(pl.String).cast(pl.Categorical))
    frame.write_parquet(temp)
    temp.replace(path)


def table_scope(path):
    if 'combined_splits' in path.parts:
        return 'combined' if path.parts[-3] in ['hitter', 'team_hitter'] else None
    stem = path.stem.split('.')[0]
    if stem.startswith('damage_pos_') or stem == 'hitter_pctiles': return 'hitter'
    return {'new_team_damage': 'team', 'new_hitting_lg_avg': 'league',
            'hitter_splits': 'hitter_splits', 'team_hitter_splits': 'team_splits',
            'hitter_pitch_splits': 'hitter_group', 'team_hitting_pitch_splits': 'team_group',
            'league_hitting_pitch_splits': 'league_group'}.get(stem)


def tracking_for(pitch, scope, path):
    team = scope.startswith('team') or (scope == 'combined' and path.parts[-3] == 'team_hitter')
    identity = ['hitting_code'] if team else [] if scope.startswith('league') else ['batter_mlbid']
    if team:
        pitch = _normalize_team_codes(pitch, 'hitting_code')
    keys = identity + BASE_KEYS
    if scope == 'combined' or scope.endswith('_group'):
        pitch = apply_game_classifications(pitch.lazy(), pl.read_parquet(CACHE)).filter(pl.col('game_type') == 'R').collect()
        pitch = add_dimensions(pitch, True)
        if team:
            pitch = pitch.with_columns(pl.col('hitting_code').replace('OAK', 'ATH'))
        if scope == 'combined':
            dims = path.parent.name.split('__')
            keys += dims
            pitch = pitch.drop_nulls(dims)
        else:
            pitch = pitch.with_columns(pl.col('split_pitch_group').replace_strict(
                {'Fastball': 'FA', 'Sinker': 'SI', 'Breaking': 'BR', 'Offspeed': 'OFF'}, default=None).alias('pitch_group'))
            keys = identity + ['season', 'level_id', 'pitch_group']
            pitch = pitch.drop_nulls('pitch_group')
    if scope.endswith('_splits'):
        parts = []
        for label, part in [
            ('vs L/R', _with_split(pitch, _split_vs_lr(pitch, 'throws'))),
            ('Home/Away', _with_split(pitch, _split_home_away(pitch, True))),
            ('Monthly', _with_split(pitch, _split_month(pitch))),
            ('1st Half/2nd Half', _split_half(pitch)),
        ]:
            parts.append(part.drop_nulls('__split').group_by(keys + ['__split']).agg(
                bat_tracking_expressions(part)).rename({'__split': 'split'}).with_columns(pl.lit(label).alias('split_type')))
        return pl.concat(parts, how='diagonal_relaxed'), keys + ['split_type', 'split']
    return pitch.group_by(keys).agg(bat_tracking_expressions(pitch)), keys


def refresh_tables(inputs, backup_root):
    paths = [p for p in OUT.rglob('*.parquet') if table_scope(p)]
    scans = [pl.scan_parquet(p) for p in inputs]
    columns = SOURCE_KEYS + FIELDS + ['season', 'level_id', 'game_type', 'game_date',
        'hitting_code', 'home_team', 'away_team', 'half_inning', 'home_mlbid', 'away_mlbid',
        'throws', 'pi_pitch_group', 'pi_pitch_type']
    changed = set()
    for scan in scans:
        for year in sorted(scan.select('season').unique().collect()['season'].to_list()):
            pitch = _add_game_type_group(scan.filter(pl.col('season') == year).select(columns).collect())
            cache = {}
            for path in paths:
                # Partition filenames let us avoid loading unrelated years.
                if '.season_' in path.name and f'.season_{year}.' not in path.name: continue
                if path.name.startswith('season_') and not path.name.startswith(f'season_{year}.'): continue
                frame = pl.read_parquet(path)
                selected = frame.filter(pl.col('season') == year)
                if selected.is_empty(): continue
                scope = table_scope(path)
                variant = (scope, path.parts[-3], path.parent.name) if scope == 'combined' else (scope,)
                if variant not in cache:
                    cache[variant] = tracking_for(pitch, scope, path)
                metrics, keys = cache[variant]
                updated = replace_tracking(selected, metrics, keys)
                # Reinsert by original row position to preserve sorting and identity.
                positions = frame.with_row_index('__position').filter(pl.col('season') == year)['__position']
                updated = updated.with_columns(positions)
                rest = frame.with_row_index('__position').filter(pl.col('season') != year)
                result = pl.concat([rest, updated], how='diagonal_relaxed').sort('__position').drop('__position')
                if not result.equals(frame):
                    write_updated(result, path, backup_root)
                    changed.add(str(path.relative_to(OUT)))
            print(f'Refreshed tracking tables: {year} ({len(changed)} files so far)', flush=True)
    return sorted(changed)


def refresh_regression(backup_root):
    from pipeline.apply_regression_from_agg import apply_mean_from_agg, round_floats
    path = OUT / 'hitters_regressed.parquet'
    frame = pl.read_parquet(path)
    years = frame['season'].unique().to_list()
    # The app's stitched hitter table includes every season represented here.
    source = OUT / f'damage_pos_{min(years)}_{max(years)}.parquet'
    hitters = pl.read_parquet(source)
    constants = pl.read_csv(ROOT / 'config/stability_constants.csv').with_columns(pl.col('season').cast(pl.Int64))
    keys = ['batter_mlbid', 'hitter_name', 'season', 'level_id', 'game_type_group']
    original = frame
    changed = []
    for stat in OUTPUT_FIELDS:
        columns = [c for c in [stat + '_raw', stat + '_reg', stat + '_n'] if c in frame.columns]
        if not columns: continue
        metrics = apply_mean_from_agg(hitters, constants, 'hitters', stat, keys)
        metrics = round_floats(metrics, 1).select(keys + columns).cast({k: frame.schema[k] for k in keys})
        frame = frame.drop(columns).join(metrics, on=keys, how='left', validate='m:1', maintain_order='left')
        changed.extend(columns)
    if len(frame) != len(original) or not frame.drop(changed).equals(original.drop(changed)):
        raise ValueError('Tracking regression changed unrelated data')
    # Restore the established column ordering.
    write_updated(frame.select(original.columns), path, backup_root)
    return str(path.relative_to(OUT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, default=ROOT / 'data/raw/bat_tracking/measurements.parquet')
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--enrich-only', action='store_true')
    parser.add_argument('--refresh-only', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    backup_root = ROOT / 'data/backups/tracking_backfill' / stamp
    backup_root.mkdir(parents=True, exist_ok=True)
    report = {'raw': [], 'updated_tables': []}
    inputs = []
    if not args.refresh_only:
        measurements = pl.read_parquet(args.measurements)
        for path in args.input:
            local = path.resolve().is_relative_to((ROOT / 'data/raw').resolve())
            destination = path if local else ROOT / 'data/raw/pitch_data_2023_2025_tracking.parquet'
            report['raw'].append(enrich_raw(path, destination, measurements, backup_root, None if local else 2023))
            inputs.append(destination)
    else:
        inputs = args.input
    if not args.enrich_only:
        report['updated_tables'] = refresh_tables(inputs, backup_root)
        report['updated_tables'].append(refresh_regression(backup_root))
    (backup_root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Backfill report: {backup_root / "report.json"}', flush=True)


if __name__ == '__main__':
    main()

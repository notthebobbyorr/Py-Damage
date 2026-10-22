"""Extend the existing scraper's bat-tracking history without altering its snapshots."""
import argparse
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['bat_speed', 'swing_length', 'attack_angle', 'attack_direction',
          'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches',
          'intercept_ball_minus_batter_pos_y_inches', 'miss_distance']
KEYS = ['game_pk', 'at_bat_number', 'pitch_number', 'batter', 'pitcher']


def compact(scan):
    return scan.select(KEYS + ['game_date', 'game_type'] + FIELDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scraper-root', type=Path)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--update', action='store_true', help='Continue the configured local scraper snapshot')
    parser.add_argument('--offline', action='store_true', help='Rebuild from saved schedule and download chunks')
    parser.add_argument('--start', type=date.fromisoformat, default=date(2026, 9, 19))
    parser.add_argument('--end', type=date.fromisoformat, default=date.today() - timedelta(days=1))
    parser.add_argument('--output', type=Path, default=ROOT / 'data/raw/bat_tracking')
    args = parser.parse_args()
    if args.update:
        config = json.loads((args.output / 'scraper_config.json').read_text(encoding='utf-8'))
        args.scraper_root = Path(config['scraper_root'])
        args.history = Path(config['history'])
        previous = json.loads((args.output / 'manifest.json').read_text(encoding='utf-8'))
        args.start = date.fromisoformat(previous['end']) + timedelta(days=1)
        if args.start > args.end:
            print(f'Scraper tracking already checked through {previous["end"]}', flush=True)
            return
    if args.scraper_root is None or args.history is None:
        parser.error('--scraper-root and --history are required for the first download')
    sys.path.insert(0, str(args.scraper_root))
    from baseball_data import BaseballClient, HttpTransport

    # Reuse the scraper's validated, capped-window bulk downloader and transport.
    source = args.scraper_root / 'sessions/2026_09_19_mlb_swing_history/pull_history.py'
    spec = importlib.util.spec_from_file_location('tracking_history_scraper', source)
    scraper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scraper)
    args.output.mkdir(parents=True, exist_ok=True)
    scraper.SESSION = args.output
    transport = HttpTransport(cache_dir=args.output / 'http_cache', timeout=120, retries=3, min_interval=1)
    client = BaseballClient(transport=transport)
    if args.offline:
        import pandas as pd
        schedule = pd.read_parquet(args.output / 'schedule.parquet')
    else:
        schedule = client.games(str(args.start), str(args.end), refresh=True).tables['games']
        schedule.to_parquet(args.output / 'schedule.parquet', index=False)
    final = schedule if schedule.empty else schedule.loc[schedule.abstract_state.eq('Final') &
                         ~schedule.status.isin(['Cancelled', 'Postponed'])]
    dates = set() if final.empty else set(final.official_date.dropna().astype(str))
    records = []
    for start, end in scraper.windows(args.start, args.end):
        if any(str(start + timedelta(days=i)) in dates for i in range((end-start).days+1)):
            records.extend(scraper.download(transport, start, end))
    parts = [compact(pl.scan_parquet(args.history))]
    parts += [compact(pl.scan_parquet(record['parquet'])) for record in records]
    frame = pl.concat(parts, how='vertical_relaxed').collect()
    if frame.select(KEYS).is_duplicated().any():
        raise ValueError('Duplicate pitch identities across scraper snapshots')
    downloaded = set(frame.filter(pl.col('game_date') >= str(args.start))['game_pk'].to_list())
    missing_games = set(final.game_pk.astype(int)) - downloaded
    if missing_games:
        raise ValueError(f'Final games absent from scraper output: {sorted(missing_games)}')
    path = args.output / 'measurements.parquet'
    temp = path.with_suffix('.tmp.parquet')
    frame.write_parquet(temp)
    temp.replace(path)
    report = {'history': str(args.history), 'start': str(args.start), 'end': str(args.end),
              'final_games': len(final), 'new_rows': sum(r['rows'] for r in records),
              'total_rows': frame.height, 'downloads': records}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'scraper_config.json').write_text(json.dumps({
        'scraper_root': str(args.scraper_root), 'history': str(path),
    }, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'downloads'}), flush=True)


if __name__ == '__main__':
    main()

"""Restore observed damage rates from preserved model-source tables, with backups."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.backfill_bat_tracking import OUT, write_updated


def main():
    backup_root = ROOT / 'data/backups/damage_rate' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    changed = 0
    for source_name, stems in [('pitcher_p_damage', ['pitcher_stuff_new', 'pitcher_pctiles']),
                               ('pitch_types_p_damage', ['new_pitch_types', 'pitch_types_pctiles'])]:
        source = pl.read_parquet(ROOT / 'data/raw' / (source_name + '.parquet'))
        keys = ['pitcher_mlbid', 'season', 'level_id', 'game_type_group']
        if source_name.startswith('pitch_types'):
            keys.append('pitch_tag')
        rates = source.group_by(keys).agg(
            ((pl.col('Damage_pct') * pl.col('n')).sum() / pl.col('n').sum()).alias('Damage_pct'))
        for path in OUT.rglob('*.parquet'):
            if path.stem.split('.')[0] not in stems:
                continue
            frame = pl.read_parquet(path)
            if 'game_type_group' not in frame:
                frame = frame.with_columns(pl.lit('Regular Season').alias('game_type_group'))
            rates_cast = rates.cast({k: frame.schema[k] for k in keys})
            result = frame.drop('Damage_pct', strict=False).join(rates_cast, on=keys,
                how='left', validate='m:1', maintain_order='left')
            preserved = [c for c in frame.columns if c != 'Damage_pct']
            assert result.select(preserved).equals(frame.select(preserved)), path
            write_updated(result, path, backup_root)
            changed += 1
    print(f'Restored Damage_pct in {changed} files; backups: {backup_root}', flush=True)


if __name__ == '__main__':
    main()

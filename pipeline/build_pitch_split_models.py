"""Backfill modeled rates and Execution Grade in existing pitching split tables."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import polars as pl
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.build_combined_splits import add_dimensions, dimension_sets
from pipeline.build_p_swstr_sources import WHIFF_BASE_NUM_FEATURES, WHIFF_CAT_FEATURES, predict_in_batches
from pipeline.combined_execution import calibrate_means
from pipeline.data_aggregate import _add_game_type_group, _tag_pitch
from pipeline.pitch_split_games import apply_game_classifications, verified_games

OUT = ROOT / "data/output"
KEYS = ["pitcher_mlbid", "pitcher_hand", "season", "level_id", "game_type_group"]
GAME_KEYS = ["game_pk", "pitcher_mlbid", "season", "level_id", "game_type"]
SPLITS = {"split_hand": "vs L/R", "split_location": "Home/Away",
          "split_half": "1st Half/2nd Half", "split_month": "Monthly"}
METRICS = {"p_SwStr_pct": "p_swstr", "p_Damage_pct": "p_damage", "grade_v13": "pred_v13"}
SUMS = [f"{metric}_{stat}" for metric in METRICS.values() for stat in ("sum", "n")]


def game_scores(raw, scored, models):
    """Reduce to sums and counts per game, pitch type, and opponent hand."""
    dims = ["split_location", "split_half", "split_month"]
    execution_keys = GAME_KEYS + ["throws"]
    meta = raw.select(execution_keys + ["pitcher_hand", *dims]).unique().with_columns(
        pl.lit(True).alias("matched_game"))
    keys = GAME_KEYS + ["pitcher_hand", *SPLITS, "pitch_tag"]
    features = WHIFF_BASE_NUM_FEATURES + WHIFF_CAT_FEATURES
    eligible = raw.filter(pl.col("pitcher_role").is_in(["SP", "RP"]))
    in_play = eligible.filter(pl.col("is_in_play").fill_null(False))
    swing, whiff, damage = models
    predictions = eligible.select(features).to_pandas()
    swstr = predict_in_batches(swing, predictions, features, 250000)
    swstr *= predict_in_batches(whiff, predictions, features, 250000)
    damage_prob = predict_in_batches(damage, in_play.select(features).to_pandas(), features, 250000)
    execution = pl.read_parquet(scored, columns=execution_keys + ["pitch_tag", "stands", "pred_v13"])
    execution = execution.cast({key: meta.schema[key] for key in execution_keys}).join(
        meta, on=execution_keys, how="left", validate="m:1", nulls_equal=True
    ).with_columns(pl.col("stands").str.to_uppercase().alias("split_hand"))
    if execution["matched_game"].null_count():
        raise ValueError(f"{scored}: execution scores have unmatched game metadata")
    parts = (
        eligible.select(keys).with_columns(pl.Series("p_swstr", swstr)),
        in_play.select(keys).with_columns(pl.Series("p_damage", damage_prob)),
        execution.select(keys + ["pred_v13"]),
    )
    return pl.concat([
        frame.group_by(keys).agg(pl.col(metric).sum().alias(metric + "_sum"),
                                 pl.col(metric).count().alias(metric + "_n"))
        for frame, metric in zip(parts, METRICS.values(), strict=True)
    ], how="diagonal_relaxed")


def aggregate(scores, dims, calibration):
    keys = KEYS + dims
    by_type = scores.drop_nulls(dims).group_by(keys + ["pitch_tag"]).agg(pl.col(SUMS).sum())
    graded = calibrate_means(by_type.with_columns(
        (pl.col("pred_v13_sum") / pl.col("pred_v13_n")).alias("pred_v13"),
        pl.col("pred_v13_n").alias("execution_n"),
    ), calibration)
    pitchers = graded.group_by(keys).agg(
        pl.col(SUMS).sum(),
        pl.col("execution_n").filter(pl.col("grade_v13").is_not_null()).sum().alias("execution_n"),
        (pl.col("grade_v13") * pl.col("execution_n")).sum().alias("weighted_grade"),
    ).with_columns(
        pl.when(pl.col("execution_n") > 0)
        .then((pl.col("weighted_grade") / pl.col("execution_n")).round(1)).alias("grade_v13")
    )
    return {
        kind: frame.with_columns([
            pl.when(pl.col(metric + "_n") > 0)
            .then(100 * pl.col(metric + "_sum") / pl.col(metric + "_n")).alias(column)
            for column, metric in METRICS.items() if column != "grade_v13"
        ]).select(keys + (["pitch_tag"] if kind == "pitch" else []) + list(METRICS))
        for kind, frame in (("pitch", graded), ("pitcher", pitchers))
    }


def stage(path, metrics, dims, stage_dir):
    original = pl.read_parquet(path)
    keys = KEYS + dims + (["pitch_tag"] if "pitch_tag" in original else [])
    # An unselected season must keep its existing model values on a daily update.
    years = metrics["season"].unique().cast(original.schema["season"]).implode()
    levels = original["level_id"].unique().cast(metrics.schema["level_id"]).implode()
    metrics = metrics.filter(pl.col("level_id").is_in(levels)).cast(
        {key: original.schema[key] for key in keys})
    selected = pl.col("season").is_in(years)
    base = original.with_columns([
        pl.col(column).cast(pl.Float32) if column in original else pl.lit(None, pl.Float32).alias(column)
        for column in METRICS
    ])
    result = base.join(metrics, on=keys, how="left", validate="m:1",
                       maintain_order="left", suffix="_new", nulls_equal=True)
    result = result.with_columns([
        pl.when(selected).then(pl.col(column + "_new").cast(pl.Float32))
        .otherwise(pl.col(column)).alias(column) for column in METRICS
    ]).select(base.columns)
    if not result.drop(list(METRICS)).equals(original.drop([c for c in METRICS if c in original])):
        raise ValueError(f"Unexpected changes to existing stats in {path}")
    staged = stage_dir / path.relative_to(OUT)
    staged.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(staged)
    return result.select([pl.col(c).count() for c in METRICS]).row(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One raw pitch file per season")
    parser.add_argument("--scored", type=Path, nargs="*", default=[], help="Existing single-season V13 scores")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    scans = [pl.scan_parquet(path) for path in args.input]
    seasons = [scan.select("season").unique().collect().item() for scan in scans]
    if len(seasons) != len(set(seasons)):
        raise ValueError("Supply exactly one raw file per selected season")
    cached_scores = {pl.scan_parquet(path).select("season").unique().collect().item(): path
                     for path in args.scored}
    verified = verified_games(scans)
    calibration = pl.read_csv(ROOT / "models/location_v13_grade_calibration.csv")
    models = [CatBoostClassifier().load_model(str(ROOT / "models" / name)) for name in
              ("is_swing_catboost_model.cbm", "p_whiff_v2.cbm", "is_damage_catboost_model.cbm")]
    columns = list(dict.fromkeys(GAME_KEYS + ["pitcher_hand", "pitcher_role", "is_in_play",
                   *WHIFF_BASE_NUM_FEATURES, *WHIFF_CAT_FEATURES, *(
                       "game_date home_team away_team half_inning home_mlbid away_mlbid "
                       "pitching_code pi_pitch_group pi_pitch_type pi_pitch_sub_type"
                   ).split()]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_dir = ROOT / "data/backups/pitch_split_models" / stamp
    with TemporaryDirectory(dir=OUT.parent) as work:
        work = Path(work)
        stage_dir = work / "stage"
        standard = {"pitch": [], "pitcher": []}
        for path, scan, year in zip(args.input, scans, seasons, strict=True):
            scored = cached_scores.get(year)
            if scored is None:
                subprocess.run([sys.executable, str(ROOT / "pipeline/location_v13_apply.py"),
                                "--data-path", str(path), "--seasons", str(year),
                                "--output-dir", str(work)], check=True)
                scored = work / "location_v13_scored.parquet"
            raw = add_dimensions(_tag_pitch(scan.select(columns).collect()), False)
            scores = game_scores(raw, scored, models)
            single = _add_game_type_group(scores)
            combined = _add_game_type_group(apply_game_classifications(scores.lazy(), verified)
                                            .filter(pl.col("game_type") == "R").collect())
            for dim, label in SPLITS.items():
                for kind, frame in aggregate(single, [dim], calibration).items():
                    if dim == "split_hand":
                        frame = frame.with_columns((pl.lit("vs ") + pl.col(dim)).alias(dim))
                    standard[kind].append(frame.rename({dim: "split"}).with_columns(
                        pl.lit(label).alias("split_type")))
            counts = {"pitch": 0, "pitcher": 0}
            for dims in dimension_sets(False):
                for kind, frame in aggregate(combined, dims, calibration).items():
                    folder = OUT / "combined_splits" / kind / "__".join(dims)
                    for partition in folder.glob(f"season_{year}.level_*.parquet"):
                        level = int(partition.stem.split("level_")[-1])
                        coverage = stage(partition, frame.filter(pl.col("level_id") == level), dims, stage_dir)
                        counts[kind] += coverage[2]
            print(f"{year}: combined rows with execution grades {counts}", flush=True)

        for kind, stem in [("pitch", "pitch_types_splits"), ("pitcher", "pitcher_splits")]:
            scores = pl.concat(standard[kind])
            path = OUT / f"{stem}.parquet"
            print(f"{path.name}: non-null {list(METRICS)} = "
                  f"{stage(path, scores, ['split_type', 'split'], stage_dir)}", flush=True)
            for year in seasons:
                chunk = OUT / "_season_chunks" / f"{stem}.season_{year}.parquet"
                if chunk.exists():
                    stage(chunk, scores.filter(pl.col("season") == year), ["split_type", "split"], stage_dir)
        if args.check_only:
            return
        staged = sorted(stage_dir.rglob("*.parquet"))
        for path in staged:
            rel = path.relative_to(stage_dir)
            backup = backup_dir / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OUT / rel, backup)
        for path in staged:
            path.replace(OUT / path.relative_to(stage_dir))
        print(f"Updated {len(staged)} tables. Backups: {backup_dir}", flush=True)


if __name__ == "__main__":
    main()

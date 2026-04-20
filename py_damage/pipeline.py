from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

try:
    from pybaseball import statcast
except ImportError:  # pragma: no cover - optional dependency
    statcast = None


IN_PLAY_DESCRIPTIONS = {
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
SWING_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
    "foul_pitchout",
    "swinging_strike",
    "swinging_strike_blocked",
    "swinging_strike_pitchout",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
CONTACT_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
    "foul_pitchout",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
CALLED_STRIKE_DESCRIPTIONS = {
    "called_strike",
    "called_strike_pitchout",
}


PITCH_TAG_MAP = {
    "SI": "SI",
    "FC": "HC",
    "FS": "FS",
    "FF": "FA",
    "FA": "FA",
    "FT": "FA",
    "SL": "SL",
    "ST": "SL",
    "SV": "SL",
    "CU": "CU",
    "KC": "CU",
    "CS": "CU",
    "CH": "CH",
}


@dataclass
class PipelineConfig:
    start_date: str
    end_date: str
    season: int
    level_id: int = 1
    output_dir: Path = Path(".")
    damage_ev_threshold: float = 95.0
    input_csv: Path | None = None
    positions_csv: Path | None = None
    output_tag: str | None = None


MISSING_COLUMNS = [
    "game_pk",
    "game_date",
    "game_type",
    "inning_topbot",
    "home_team",
    "away_team",
    "batter",
    "pitcher",
    "stand",
    "p_throws",
    "events",
    "description",
    "type",
    "launch_speed",
    "launch_angle",
    "hc_x",
    "hc_y",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "pitch_type",
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "outs_on_play",
    "at_bat_number",
    "player_name",
    "batter_name",
]


POSITION_COLUMNS = ["UT", "C", "X1B", "X2B", "X3B", "SS", "OF", "P"]


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Build Damage data from Statcast.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--level-id", type=int, default=1)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--damage-ev-threshold", type=float, default=95.0)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--positions-csv", default=None)
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    return PipelineConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        season=args.season,
        level_id=args.level_id,
        output_dir=Path(args.output_dir),
        damage_ev_threshold=args.damage_ev_threshold,
        input_csv=Path(args.input_csv) if args.input_csv else None,
        positions_csv=Path(args.positions_csv) if args.positions_csv else None,
        output_tag=args.output_tag,
    )


def ensure_columns(df: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if not missing:
        return df
    return df.with_columns([pl.lit(None).alias(col) for col in missing])


def load_raw_data(config: PipelineConfig) -> pl.DataFrame:
    if config.input_csv:
        df = pl.read_csv(config.input_csv)
    else:
        if statcast is None:
            raise RuntimeError("pybaseball is required when --input-csv is not provided")
        raw = statcast(config.start_date, config.end_date)
        df = pl.from_pandas(raw)
    return ensure_columns(df, MISSING_COLUMNS)


def add_pitch_features(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("game_date").str.strptime(pl.Date, strict=False).alias("game_date"),
            pl.col("stand").alias("batter_hand"),
            pl.col("p_throws").alias("pitcher_hand"),
            pl.col("batter").alias("batter_mlbid"),
            pl.col("pitcher").alias("pitcher_mlbid"),
            pl.col("launch_speed").alias("exit_velo"),
            pl.col("release_speed").alias("pitch_velo"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("inning_topbot") == "Bot")
            .then(pl.col("home_team"))
            .otherwise(pl.col("away_team"))
            .alias("hitting_code"),
            pl.when(pl.col("inning_topbot") == "Bot")
            .then(pl.col("away_team"))
            .otherwise(pl.col("home_team"))
            .alias("pitching_code"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("batter_name").is_not_null())
            .then(pl.col("batter_name"))
            .otherwise(pl.col("batter").cast(pl.Utf8))
            .alias("hitter_name"),
            pl.when(pl.col("player_name").is_not_null())
            .then(pl.col("player_name"))
            .otherwise(pl.col("pitcher").cast(pl.Utf8))
            .alias("pitcher_name"),
        ]
    )

    df = df.with_columns(
        [
            pl.lit(config.level_id).alias("level_id"),
            pl.lit(config.season).alias("season"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("hc_x").is_not_null() & pl.col("hc_y").is_not_null())
            .then(pl.col("hc_x") - 125.42)
            .otherwise(None)
            .alias("hc_x_adj"),
            pl.when(pl.col("hc_y").is_not_null())
            .then(198.27 - pl.col("hc_y"))
            .otherwise(None)
            .alias("hc_y_adj"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("hc_x_adj").is_not_null() & (pl.col("hc_y_adj") != 0))
            .then((pl.col("hc_x_adj") / pl.col("hc_y_adj")).arctan() * (180 / 3.14159265 * 0.75))
            .otherwise(None)
            .alias("spray_angle"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("batter_hand") == "L")
            .then(-1 * pl.col("spray_angle"))
            .otherwise(pl.col("spray_angle"))
            .alias("spray_angle_adj"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("description").is_in(IN_PLAY_DESCRIPTIONS).alias("is_in_play"),
            pl.col("description").is_in(SWING_DESCRIPTIONS).alias("swing"),
            pl.col("description").is_in(CONTACT_DESCRIPTIONS).alias("is_contact"),
            pl.col("description").is_in(CALLED_STRIKE_DESCRIPTIONS).alias("called_strike"),
            (pl.col("type") == "B").alias("is_ball"),
        ]
    )

    df = df.with_columns(
        [
            (
                (pl.col("plate_x").is_not_null())
                & (pl.col("plate_z").is_not_null())
                & (pl.col("sz_bot").is_not_null())
                & (pl.col("sz_top").is_not_null())
                & (pl.col("plate_x").abs() <= 0.83)
                & (pl.col("plate_z") >= pl.col("sz_bot"))
                & (pl.col("plate_z") <= pl.col("sz_top"))
            ).alias("is_inzone"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("pitch_type").is_in(list(PITCH_TAG_MAP.keys())))
            .then(pl.col("pitch_type").replace(PITCH_TAG_MAP))
            .otherwise("XX")
            .alias("pitch_tag"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("pitch_tag").is_in(["FA", "SI", "HC"]))
            .then("FA")
            .when(pl.col("pitch_tag").is_in(["SL", "SW", "CU"]))
            .then("BR")
            .when(pl.col("pitch_tag").is_in(["CH", "FS"]))
            .then("OFF")
            .otherwise("OTHER")
            .alias("pitch_group"),
        ]
    )

    if "decision_value" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("decision_value"))

    df = df.with_columns(
        [
            (
                (pl.col("exit_velo") >= config.damage_ev_threshold)
                & (pl.col("launch_angle") > 0)
                & (pl.col("spray_angle_adj") >= -50)
                & (pl.col("spray_angle_adj") <= 50)
            ).alias("damage"),
            (pl.col("swing") & (~pl.col("is_contact"))).alias("whiff"),
        ]
    )

    df = df.with_columns(
        [
            (1.8 * pl.col("pfx_z")).alias("ivb"),
            (1.8 * pl.col("pfx_x")).alias("hb"),
            (1.8 * pl.col("pfx_x")).alias("hb_orig"),
        ]
    )

    df = df.with_columns(
        [
            pl.concat_str(
                [pl.col("game_pk").cast(pl.Utf8), pl.col("at_bat_number").cast(pl.Utf8)],
                separator="-",
            ).alias("pa_id"),
        ]
    )

    return df


def safe_rate(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator > 0).then((numerator / denominator) * 100).otherwise(None)


def build_hitter_table(df: pl.DataFrame, positions_df: pl.DataFrame | None) -> pl.DataFrame:
    base = (
        df.group_by(["batter_mlbid", "hitter_name", "level_id", "hitting_code", "season"])
        .agg(
            [
                pl.count().alias("pitches"),
                pl.col("pa_id").n_unique().alias("PA"),
                pl.col("is_in_play").sum().alias("bbe"),
                pl.col("damage").sum().alias("damage_ct"),
                pl.col("exit_velo").filter(pl.col("is_in_play") & (pl.col("exit_velo") > 0)).quantile(0.9).alias("EV90th"),
                pl.col("exit_velo").filter(pl.col("is_in_play") & (pl.col("exit_velo") > 0)).max().alias("max_EV"),
                pl.sum(
                    (pl.col("launch_angle") > 20)
                    & (pl.col("spray_angle_adj") < -15)
                    & pl.col("is_in_play")
                ).alias("pull_fb"),
                pl.sum(pl.col("swing") & (~pl.col("is_inzone"))).alias("chase_swings"),
                pl.sum(~pl.col("is_inzone")).alias("out_zone"),
                pl.sum(pl.col("swing") & pl.col("is_inzone") & pl.col("is_contact")).alias("zone_contact"),
                pl.sum(pl.col("swing") & pl.col("is_inzone")).alias("zone_swings"),
                pl.sum(pl.col("whiff") & (pl.col("pitch_group") != "FA")).alias("secondary_whiffs"),
                pl.sum(pl.col("swing") & (pl.col("pitch_group") != "FA")).alias("secondary_swings"),
                pl.sum(pl.col("is_contact")).alias("contact_ct"),
                pl.sum(pl.col("swing")).alias("swing_ct"),
                pl.sum((pl.col("decision_value") > 0) & (pl.col("swing") == False)).alias("good_takes"),
                pl.sum(pl.col("decision_value") > 0).alias("good_decisions"),
                pl.sum((pl.col("decision_value") < 0) & (pl.col("swing") == False)).alias("bad_takes"),
                pl.sum(pl.col("swing") == False).alias("takes"),
            ]
        )
        .with_columns(
            [
                safe_rate(pl.col("damage_ct"), pl.col("bbe")).alias("damage_rate"),
                safe_rate(pl.col("pull_fb"), pl.col("bbe")).alias("pull_FB_pct"),
                safe_rate(pl.col("chase_swings"), pl.col("out_zone")).alias("chase"),
                safe_rate(pl.col("zone_contact"), pl.col("zone_swings")).alias("z_con"),
                safe_rate(pl.col("secondary_whiffs"), pl.col("secondary_swings")).alias("secondary_whiff_pct"),
                safe_rate(pl.col("good_takes"), pl.col("good_decisions")).alias("selection_skill"),
                safe_rate(pl.col("bad_takes"), pl.col("takes")).alias("hittable_pitches_taken"),
            ]
        )
        .with_columns((pl.col("selection_skill") - pl.col("hittable_pitches_taken")).alias("SEAGER"))
    )

    league_contact = (
        base.group_by(["season", "level_id"])
        .agg((pl.col("contact_ct").sum() / pl.col("swing_ct").sum()).alias("league_contact"))
    )

    base = base.join(league_contact, on=["season", "level_id"], how="left").with_columns(
        ((pl.col("contact_ct") / pl.col("swing_ct") - pl.col("league_contact")) * 100).alias(
            "contact_vs_avg"
        )
    )

    if positions_df is None:
        base = base.with_columns([pl.lit(0).alias(col) for col in POSITION_COLUMNS])
    else:
        base = base.join(positions_df, on="batter_mlbid", how="left")
        for col in POSITION_COLUMNS:
            if col not in base.columns:
                base = base.with_columns(pl.lit(0).alias(col))

    return (
        base.select(
            POSITION_COLUMNS
            + [
                "batter_mlbid",
                "hitter_name",
                "level_id",
                "hitting_code",
                "season",
                "pitches",
                "PA",
                "bbe",
                "damage_rate",
                "EV90th",
                "max_EV",
                "pull_FB_pct",
                "SEAGER",
                "selection_skill",
                "hittable_pitches_taken",
                "chase",
                "z_con",
                "secondary_whiff_pct",
                "contact_vs_avg",
            ]
        )
    )


def build_team_hitting(hitters: pl.DataFrame) -> pl.DataFrame:
    return (
        hitters.group_by(["hitting_code", "level_id", "season"])
        .agg(
            [
                pl.col("PA").sum().alias("PA"),
                pl.col("bbe").sum().alias("bbe"),
                pl.col("damage_rate").mean().alias("damage_rate"),
                pl.col("EV90th").mean().alias("EV90th"),
                pl.col("pull_FB_pct").mean().alias("pull_FB_pct"),
                pl.col("SEAGER").mean().alias("SEAGER"),
                pl.col("selection_skill").mean().alias("selection_skill"),
                pl.col("hittable_pitches_taken").mean().alias("hittable_pitches_taken"),
                pl.col("chase").mean().alias("chase"),
                pl.col("z_con").mean().alias("z_con"),
                pl.col("contact_vs_avg").mean().alias("contact_vs_avg"),
                pl.col("secondary_whiff_pct").mean().alias("secondary_whiff_pct"),
            ]
        )
    )


def build_hitting_lg_avg(hitters: pl.DataFrame) -> pl.DataFrame:
    return (
        hitters.group_by(["level_id", "season"])
        .agg(
            [
                pl.col("PA").sum().alias("PA"),
                pl.col("bbe").sum().alias("bbe"),
                pl.col("damage_rate").mean().alias("damage_rate"),
                pl.col("EV90th").mean().alias("EV90th"),
                pl.col("pull_FB_pct").mean().alias("pull_FB_pct"),
                pl.col("SEAGER").mean().alias("SEAGER"),
                pl.col("selection_skill").mean().alias("selection_skill"),
                pl.col("hittable_pitches_taken").mean().alias("hittable_pitches_taken"),
                pl.col("chase").mean().alias("chase"),
                pl.col("z_con").mean().alias("z_con"),
                pl.col("contact_vs_avg").mean().alias("contact_vs_avg"),
            ]
        )
    )


def build_pitcher_table(df: pl.DataFrame) -> pl.DataFrame:
    base = (
        df.group_by(["pitcher_mlbid", "pitcher_name", "level_id", "season", "pitching_code", "pitcher_hand"])
        .agg(
            [
                pl.count().alias("pitches"),
                pl.col("pa_id").n_unique().alias("TBF"),
                pl.col("game_pk").n_unique().alias("G"),
                pl.col("outs_on_play").sum().alias("outs_recorded"),
                pl.sum(pl.col("whiff")).alias("whiff_ct"),
                pl.sum(pl.col("called_strike")).alias("called_strike_ct"),
                pl.sum(pl.col("is_ball")).alias("ball_ct"),
                pl.sum(pl.col("swing") & pl.col("is_inzone") & pl.col("is_contact")).alias("zone_contact"),
                pl.sum(pl.col("swing") & pl.col("is_inzone")).alias("zone_swings"),
                pl.sum(pl.col("swing") & (~pl.col("is_inzone"))).alias("chase_swings"),
                pl.sum(~pl.col("is_inzone")).alias("out_zone"),
                pl.col("pitch_velo").filter(pl.col("pitch_group") == "FA").mean().alias("fastball_velo"),
                pl.col("pitch_velo").max().alias("max_velo"),
                pl.col("release_pos_z").mean().alias("rel_z"),
                pl.col("release_pos_x").mean().alias("rel_x"),
                pl.col("release_extension").mean().alias("ext"),
            ]
        )
        .with_columns(
            [
                (pl.col("outs_recorded") / 3).alias("IP"),
                safe_rate(pl.col("whiff_ct"), pl.col("pitches")).alias("SwStr"),
                safe_rate(pl.col("ball_ct"), pl.col("pitches")).alias("Ball_pct"),
                safe_rate(pl.col("zone_contact"), pl.col("zone_swings")).alias("Z_Contact"),
                safe_rate(pl.col("chase_swings"), pl.col("out_zone")).alias("Chase"),
                safe_rate(pl.col("whiff_ct") + pl.col("called_strike_ct"), pl.col("pitches")).alias("CSW"),
                (pl.col("TBF") / pl.col("G")).alias("TBF_per_G"),
            ]
        )
        .with_columns(
            [
                pl.lit(None).alias("std.ZQ"),
                pl.lit(None).alias("std.DMG"),
                pl.lit(None).alias("std.NRV"),
                pl.lit(None).alias("fastball_vaa"),
            ]
        )
    )

    return base.select(
        [
            "level_id",
            "TBF_per_G",
            "pitcher_hand",
            "pitcher_mlbid",
            pl.col("pitcher_name").alias("name"),
            "season",
            "pitching_code",
            "pitches",
            "TBF",
            "IP",
            "std.ZQ",
            "std.DMG",
            "std.NRV",
            "fastball_velo",
            "max_velo",
            "fastball_vaa",
            "SwStr",
            "Ball_pct",
            "Z_Contact",
            "Chase",
            "CSW",
            "rel_z",
            "rel_x",
            "ext",
        ]
    )


def build_team_pitching(pitchers: pl.DataFrame) -> pl.DataFrame:
    return (
        pitchers.group_by(["level_id", "season", "pitching_code"])
        .agg(
            [
                pl.col("IP").sum().alias("IP"),
                pl.col("std.ZQ").mean().alias("std.ZQ"),
                pl.col("std.DMG").mean().alias("std.DMG"),
                pl.col("std.NRV").mean().alias("std.NRV"),
                pl.col("fastball_velo").mean().alias("fastball_velo"),
                pl.col("fastball_vaa").mean().alias("fastball_vaa"),
                pl.col("SwStr").mean().alias("SwStr"),
                pl.col("Ball_pct").mean().alias("Ball_pct"),
                pl.col("Z_Contact").mean().alias("Z_Contact"),
                pl.col("Chase").mean().alias("Chase"),
                pl.col("CSW").mean().alias("CSW"),
            ]
        )
    )


def build_pitching_lg_avg(pitchers: pl.DataFrame) -> pl.DataFrame:
    return (
        pitchers.group_by(["level_id", "season"])
        .agg(
            [
                pl.col("fastball_velo").mean().alias("fastball_velo"),
                pl.col("fastball_vaa").mean().alias("fastball_vaa"),
                pl.col("SwStr").mean().alias("SwStr"),
                pl.col("Ball_pct").mean().alias("Ball_pct"),
                pl.col("Z_Contact").mean().alias("Z_Contact"),
                pl.col("Chase").mean().alias("Chase"),
                pl.col("CSW").mean().alias("CSW"),
            ]
        )
    )


def build_pitch_types(df: pl.DataFrame) -> pl.DataFrame:
    totals = df.group_by(["pitcher_mlbid", "level_id", "season"]).agg(pl.count().alias("total_pitches"))
    base = (
        df.group_by(
            [
                "pitcher_mlbid",
                "pitcher_name",
                "pitcher_hand",
                "pitching_code",
                "season",
                "level_id",
                "pitch_tag",
            ]
        )
        .agg(
            [
                pl.count().alias("pitches"),
                pl.col("pitch_velo").mean().alias("velo"),
                pl.col("pitch_velo").max().alias("max_velo"),
                pl.col("ivb").mean().alias("ivb"),
                pl.col("hb").mean().alias("hb"),
                pl.sum(pl.col("whiff")).alias("whiff_ct"),
                pl.sum(pl.col("called_strike")).alias("called_strike_ct"),
                pl.sum(pl.col("is_ball")).alias("ball_ct"),
                pl.sum(pl.col("swing") & pl.col("is_inzone") & pl.col("is_contact")).alias("zone_contact"),
                pl.sum(pl.col("swing") & pl.col("is_inzone")).alias("zone_swings"),
                pl.sum(pl.col("swing") & (~pl.col("is_inzone"))).alias("chase_swings"),
                pl.sum(~pl.col("is_inzone")).alias("out_zone"),
                pl.sum(pl.col("is_inzone")).alias("zone_ct"),
            ]
        )
        .join(totals, on=["pitcher_mlbid", "level_id", "season"], how="left")
        .with_columns(
            [
                safe_rate(pl.col("whiff_ct"), pl.col("pitches")).alias("SwStr"),
                safe_rate(pl.col("ball_ct"), pl.col("pitches")).alias("Ball_pct"),
                safe_rate(pl.col("zone_contact"), pl.col("zone_swings")).alias("Z_Contact"),
                safe_rate(pl.col("chase_swings"), pl.col("out_zone")).alias("Chase"),
                safe_rate(pl.col("called_strike_ct") + pl.col("whiff_ct"), pl.col("pitches")).alias("CSW"),
                safe_rate(pl.col("zone_ct"), pl.col("pitches")).alias("Zone"),
                (pl.col("pitches") / pl.col("total_pitches") * 100).alias("pct"),
                pl.lit(None).alias("std.ZQ"),
                pl.lit(None).alias("std.DMG"),
                pl.lit(None).alias("std.NRV"),
                pl.lit(None).alias("vaa"),
                pl.lit(None).alias("haa"),
            ]
        )
    )

    return base.select(
        [
            pl.col("pitcher_name").alias("name"),
            "level_id",
            "pitcher_mlbid",
            "pitcher_hand",
            "pitching_code",
            "season",
            "pitches",
            "pitch_tag",
            "pct",
            "std.ZQ",
            "std.DMG",
            "std.NRV",
            "velo",
            "max_velo",
            "vaa",
            "haa",
            "ivb",
            "hb",
            "SwStr",
            "Z_Contact",
            "Ball_pct",
            "Zone",
            "Chase",
            "CSW",
        ]
    )


def add_percentiles(
    df: pl.DataFrame, group_cols: list[str], metric_cols: list[str], suffix: str = "_pctile"
) -> pl.DataFrame:
    result = df
    for metric in metric_cols:
        rank = pl.col(metric).rank(method="average").over(group_cols)
        count = pl.count().over(group_cols)
        pct = pl.when(count > 1).then((rank - 1) / (count - 1) * 100).otherwise(100)
        result = result.with_columns(pct.alias(f"{metric}{suffix}"))
    return result


def build_hitter_pctiles(hitters: pl.DataFrame) -> pl.DataFrame:
    metric_cols = [
        "SEAGER",
        "selection_skill",
        "hittable_pitches_taken",
        "damage_rate",
        "EV90th",
        "max_EV",
        "pull_FB_pct",
        "chase",
        "z_con",
        "secondary_whiff_pct",
        "contact_vs_avg",
    ]
    pct = add_percentiles(hitters, ["season", "level_id"], metric_cols)
    return pct.select(
        [
            "batter_mlbid",
            "hitter_name",
            "season",
            "level_id",
            "hitting_code",
            "SEAGER_pctile",
            "selection_skill_pctile",
            "hittable_pitches_taken_pctile",
            "damage_rate_pctile",
            "EV90th_pctile",
            "max_EV_pctile",
            "pull_FB_pct_pctile",
            "chase_pctile",
            "z_con_pctile",
            "secondary_whiff_pct_pctile",
            "contact_vs_avg_pctile",
            "pitches",
            "bbe",
            "PA",
        ]
    ).rename(
        {
            "selection_skill_pctile": "selection_pctile",
            "hittable_pitches_taken_pctile": "hittable_pitches_pctile",
            "damage_rate_pctile": "damage_pctile",
            "EV90th_pctile": "EV90_pctile",
            "max_EV_pctile": "max_pctile",
            "pull_FB_pct_pctile": "pfb_pctile",
            "secondary_whiff_pct_pctile": "sec_whiff_pctile",
            "contact_vs_avg_pctile": "c_vs_avg_pctile",
        }
    )


def build_pitcher_pctiles(pitchers: pl.DataFrame) -> pl.DataFrame:
    metric_cols = [
        "std.ZQ",
        "std.DMG",
        "std.NRV",
        "fastball_velo",
        "max_velo",
        "fastball_vaa",
        "SwStr",
        "Ball_pct",
        "Z_Contact",
        "Chase",
        "CSW",
        "rel_z",
        "rel_x",
        "ext",
    ]
    pct = add_percentiles(pitchers, ["season", "level_id"], metric_cols)
    return pct.select(
        [
            "pitcher_mlbid",
            "name",
            "season",
            "level_id",
            "pitching_code",
            "std.ZQ_pctile",
            "std.DMG_pctile",
            "std.NRV_pctile",
            "fastball_velo_pctile",
            "max_velo_pctile",
            "fastball_vaa_pctile",
            "SwStr_pctile",
            "Ball_pct_pctile",
            "Z_Contact_pctile",
            "Chase_pctile",
            "CSW_pctile",
            "rel_z_pctile",
            "rel_x_pctile",
            "ext_pctile",
            "TBF",
            "IP",
        ]
    ).rename(
        {
            "std.ZQ_pctile": "PQ_pctile",
            "std.DMG_pctile": "DMG_pctile",
            "std.NRV_pctile": "NRV_pctile",
            "fastball_velo_pctile": "FA_velo_pctile",
            "max_velo_pctile": "FA_max_pctile",
            "fastball_vaa_pctile": "FA_vaa_pctile",
            "Ball_pct_pctile": "Ball_pctile",
            "Z_Contact_pctile": "Z_con_pctile",
            "rel_z_pctile": "rZ_pctile",
            "rel_x_pctile": "rX_pctile",
        }
    )


def build_pitch_type_pctiles(pitch_types: pl.DataFrame) -> pl.DataFrame:
    metric_cols = [
        "pct",
        "std.ZQ",
        "std.DMG",
        "std.NRV",
        "velo",
        "max_velo",
        "vaa",
        "haa",
        "ivb",
        "hb",
        "SwStr",
        "Ball_pct",
        "Zone",
        "Z_Contact",
        "Chase",
        "CSW",
    ]
    pct = add_percentiles(pitch_types, ["season", "level_id", "pitch_tag"], metric_cols)
    return pct.select(
        [
            "pitcher_mlbid",
            "name",
            "season",
            "level_id",
            "pitching_code",
            "pitch_tag",
            "pitcher_hand",
            "pct_pctile",
            "std.ZQ_pctile",
            "std.DMG_pctile",
            "std.NRV_pctile",
            "velo_pctile",
            "max_velo_pctile",
            "vaa_pctile",
            "haa_pctile",
            "ivb_pctile",
            "hb_pctile",
            "SwStr_pctile",
            "Ball_pct_pctile",
            "Zone_pctile",
            "Z_Contact_pctile",
            "Chase_pctile",
            "CSW_pctile",
            "pitches",
        ]
    ).rename(
        {
            "pct_pctile": "usage_pctile",
            "std.ZQ_pctile": "PQ_pctile",
            "std.DMG_pctile": "DMG_pctile",
            "std.NRV_pctile": "NRV_pctile",
            "Ball_pct_pctile": "Ball_pctile",
            "Zone_pctile": "zone_pctile",
            "Z_Contact_pctile": "Z_con_pctile",
        }
    )


def load_positions(config: PipelineConfig) -> pl.DataFrame | None:
    if not config.positions_csv:
        return None
    return pl.read_csv(config.positions_csv)


def output_tag(config: PipelineConfig) -> str:
    if config.output_tag:
        return config.output_tag
    return f"{config.season}"


def write_outputs(config: PipelineConfig, outputs: dict[str, pl.DataFrame]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.write_csv(config.output_dir / name)


def main() -> None:
    config = parse_args()
    raw = load_raw_data(config)
    raw = add_pitch_features(raw, config)
    positions = load_positions(config)

    hitters = build_hitter_table(raw, positions)
    team_hitting = build_team_hitting(hitters)
    hitting_avg = build_hitting_lg_avg(hitters)
    hitter_pctiles = build_hitter_pctiles(hitters)

    pitchers = build_pitcher_table(raw)
    team_pitching = build_team_pitching(pitchers)
    pitching_avg = build_pitching_lg_avg(pitchers)
    pitcher_pctiles = build_pitcher_pctiles(pitchers)

    pitch_types = build_pitch_types(raw)
    pitch_types_pctiles = build_pitch_type_pctiles(pitch_types)

    tag = output_tag(config)

    outputs = {
        f"damage_pos_{tag}.csv": hitters,
        "new_team_damage.csv": team_hitting,
        "new_hitting_lg_avg.csv": hitting_avg,
        "hitter_pctiles.csv": hitter_pctiles,
        "pitcher_stuff_new.csv": pitchers,
        "new_team_stuff.csv": team_pitching,
        "new_lg_stuff.csv": pitching_avg,
        "pitcher_pctiles.csv": pitcher_pctiles,
        "new_pitch_types.csv": pitch_types,
        "pitch_types_pctiles.csv": pitch_types_pctiles,
    }

    write_outputs(config, outputs)


if __name__ == "__main__":
    main()

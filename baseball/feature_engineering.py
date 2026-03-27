import pandas as pd
import numpy as np
from pybaseball import playerid_reverse_lookup


TOTAL_BASES_MAP = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
AB_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
    "field_out",
    "grounded_into_double_play",
    "force_out",
    "double_play",
    "fielders_choice_out",
    "fielders_choice",
    "strikeout",
    "strikeout_double_play",
    "other_out",
    "triple_play",
}


def normalize_team_abbr(team: str, game_date: pd.Timestamp | None = None) -> str | None:
    if pd.isna(team):
        return None

    team = str(team).strip().upper()
    year = pd.Timestamp(game_date).year if game_date is not None and not pd.isna(game_date) else None

    if team == "AZ":
        return "ARI"
    if team == "ATH":
        if year is not None and year <= 2024:
            return "OAK"
        return "ATH"

    return team

def add_batter_names(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()

    batter_ids = (
        pd.Series(game_df["batter"].dropna().unique())
        .astype(int)
        .tolist()
    )

    lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")

    lookup["player_name"] = (
        lookup["name_first"].fillna("").str.strip() + " " +
        lookup["name_last"].fillna("").str.strip()
    ).str.strip()

    lookup = lookup.rename(columns={"key_mlbam": "batter"})
    lookup["batter"] = lookup["batter"].astype(int)
    game_df["batter"] = game_df["batter"].astype(int)

    # Remove any existing player_name column before merge
    if "player_name" in game_df.columns:
        game_df = game_df.drop(columns=["player_name"])

    game_df = game_df.merge(
        lookup[["batter", "player_name"]],
        on="batter",
        how="left",
    )

    return game_df

def add_contact_flags(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["game_date"] = pd.to_datetime(data["game_date"])
    data["home_team"] = data.apply(
        lambda row: normalize_team_abbr(row["home_team"], row["game_date"]),
        axis=1,
    )
    data["away_team"] = data.apply(
        lambda row: normalize_team_abbr(row["away_team"], row["game_date"]),
        axis=1,
    )

    data["is_hr"] = (data["events"] == "home_run").astype(int)
    data["is_hit"] = data["events"].isin(HIT_EVENTS).astype(int)
    data["is_ab"] = data["events"].isin(AB_EVENTS).astype(int)
    data["is_pa"] = data["events"].notna().astype(int)

    data["total_bases"] = data["events"].map(TOTAL_BASES_MAP).fillna(0).astype(int)

    data["is_barrel"] = (
        (data["launch_speed"] >= 98) &
        (data["launch_angle"].between(26, 30))
    ).fillna(False).astype(int)

    data["is_hard_hit"] = (data["launch_speed"] >= 95).fillna(False).astype(int)
    data["is_fly_ball"] = (data["bb_type"] == "fly_ball").fillna(False).astype(int)

    data["estimated_slg"] = pd.to_numeric(
        data["estimated_slg_using_speedangle"], errors="coerce"
    )

    data["pull_flag"] = _build_pull_flag(data)
    data["batting_team"] = _build_batting_team(data)

    return data


def _build_pull_flag(data: pd.DataFrame) -> pd.Series:
    has_location = data["hc_x"].notna()

    pull_r = (data["stand"] == "R") & has_location & (data["hc_x"] < 125)
    pull_l = (data["stand"] == "L") & has_location & (data["hc_x"] > 125)

    return (pull_r | pull_l).astype(int)


def _build_batting_team(data: pd.DataFrame) -> pd.Series:
    """
    Infer batting team from inning_topbot.
    Top -> away team batting
    Bottom -> home team batting
    """
    batting_team = np.where(
        data["inning_topbot"] == "Top",
        data["away_team"],
        data["home_team"]
    )
    return pd.Series(batting_team, index=data.index)


def add_lineup_position(data: pd.DataFrame) -> pd.DataFrame:
    """
    Infer lineup position from first plate appearance order within each team-game.
    This is best-effort from Statcast, useful for modeling/backtesting.
    """
    data = data.copy()

    pa_rows = data[data["is_pa"] == 1].copy()

    # Use game_pk if present, otherwise fall back to game_date + teams
    game_key_cols = ["game_date", "home_team", "away_team"]
    if "game_pk" in pa_rows.columns:
        game_key_cols = ["game_pk"]

    lineup_source = (
        pa_rows.groupby(game_key_cols + ["batting_team", "batter"])
        .agg(
            first_pa_num=("at_bat_number", "min")
        )
        .reset_index()
    )

    lineup_source = lineup_source.sort_values(
        game_key_cols + ["batting_team", "first_pa_num", "batter"]
    )

    lineup_source["lineup_position"] = (
        lineup_source.groupby(game_key_cols + ["batting_team"])["first_pa_num"]
        .rank(method="dense")
        .astype(int)
    )

    lineup_cols = game_key_cols + ["batting_team", "batter", "lineup_position"]

    data = data.merge(
        lineup_source[lineup_cols],
        on=game_key_cols + ["batting_team", "batter"],
        how="left"
    )

    return data


def add_scheduled_game_number(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()

    if "game_pk" not in game_df.columns:
        game_df["scheduled_game_no"] = 1
        return game_df

    game_df = game_df.sort_values(["game_date", "home_team", "away_team", "game_pk", "batter"])
    game_level_keys = ["game_date", "home_team", "away_team", "game_pk"]

    game_order = (
        game_df[game_level_keys]
        .drop_duplicates()
        .sort_values(["game_date", "home_team", "away_team", "game_pk"])
    )

    game_order["scheduled_game_no"] = (
        game_order.groupby(["game_date", "home_team", "away_team"])
        .cumcount() + 1
    )

    game_df = game_df.merge(
        game_order,
        on=game_level_keys,
        how="left",
    )

    return game_df


def build_game_dataset(data: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["game_date", "batter"]

    keep_game_pk = "game_pk" in data.columns
    if keep_game_pk:
        group_cols.append("game_pk")

    agg_dict = {
        "hr": ("is_hr", "sum"),
        "hits": ("is_hit", "sum"),
        "ab": ("is_ab", "sum"),
        "pa": ("is_pa", "sum"),
        "total_bases": ("total_bases", "sum"),
        "avg_launch_speed": ("launch_speed", "mean"),
        "max_launch_speed": ("launch_speed", "max"),
        "avg_launch_angle": ("launch_angle", "mean"),
        "avg_distance": ("hit_distance_sc", "mean"),
        "avg_estimated_slg": ("estimated_slg", "mean"),
        "barrels": ("is_barrel", "mean"),
        "hard_hit": ("is_hard_hit", "mean"),
        "fly_ball_rate": ("is_fly_ball", "mean"),
        "pull_rate": ("pull_flag", "mean"),
        "player_name": ("player_name", "first"),
        "stand": ("stand", "first"),
        "pitcher": ("pitcher", "first"),
        "p_throws": ("p_throws", "first"),
        "home_team": ("home_team", "first"),
        "away_team": ("away_team", "first"),
        "batting_team": ("batting_team", "first"),
    }
    if "lineup_position" in data.columns:
        agg_dict["lineup_position"] = ("lineup_position", "first")

    game_df = (
        data.groupby(group_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    game_df["hr_game"] = (game_df["hr"] > 0).astype(int)

    game_df["slg"] = np.where(
        game_df["ab"] > 0,
        game_df["total_bases"] / game_df["ab"],
        np.nan,
    )

    game_df["avg"] = np.where(
        game_df["ab"] > 0,
        game_df["hits"] / game_df["ab"],
        np.nan,
    )

    game_df["iso"] = game_df["slg"] - game_df["avg"]

    game_df["hr_rate"] = np.where(
        game_df["pa"] > 0,
        game_df["hr"] / game_df["pa"],
        np.nan,
    )

    if "lineup_position" in game_df.columns:
        game_df["started"] = game_df["lineup_position"].le(9).astype(int)
    else:
        game_df["started"] = np.nan

    return game_df


def add_hitter_rolling_features(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()
    sort_cols = ["batter", "game_date"]
    if "game_pk" in game_df.columns:
        sort_cols.append("game_pk")
    if "scheduled_game_no" in game_df.columns:
        sort_cols.append("scheduled_game_no")

    game_df = game_df.sort_values(sort_cols)
    game_df["season"] = pd.to_datetime(game_df["game_date"]).dt.year

    batter_group = game_df.groupby("batter")

    game_df["rolling_hr_15"] = batter_group["hr"].transform(
        lambda x: x.shift().rolling(15, min_periods=5).mean()
    )

    game_df["rolling_hr_30"] = batter_group["hr"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    game_df["rolling_barrel_30"] = batter_group["barrels"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    game_df["rolling_hardhit_30"] = batter_group["hard_hit"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    game_df["rolling_ev_30"] = batter_group["avg_launch_speed"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    game_df["rolling_max_ev_30"] = batter_group["max_launch_speed"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).max()
    )

    game_df["rolling_pull_rate_30"] = batter_group["pull_rate"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    game_df["rolling_flyball_rate_30"] = batter_group["fly_ball_rate"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    game_df["rolling_xslg_30"] = batter_group["avg_estimated_slg"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    rolling_hr_sum_30 = batter_group["hr"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).sum()
    )
    rolling_pa_sum_30 = batter_group["pa"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).sum()
    )
    game_df["rolling_hr_rate_30"] = np.where(
        rolling_pa_sum_30 > 0,
        rolling_hr_sum_30 / rolling_pa_sum_30,
        np.nan,
    )

    rolling_tb_sum_30 = batter_group["total_bases"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).sum()
    )
    rolling_hits_sum_30 = batter_group["hits"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).sum()
    )
    rolling_ab_sum_30 = batter_group["ab"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).sum()
    )

    rolling_slg_30 = np.where(
        rolling_ab_sum_30 > 0,
        rolling_tb_sum_30 / rolling_ab_sum_30,
        np.nan,
    )
    rolling_avg_30 = np.where(
        rolling_ab_sum_30 > 0,
        rolling_hits_sum_30 / rolling_ab_sum_30,
        np.nan,
    )

    game_df["rolling_slg_30"] = rolling_slg_30
    game_df["rolling_avg_30"] = rolling_avg_30
    game_df["rolling_iso_30"] = rolling_slg_30 - rolling_avg_30

    career_hr_sum = batter_group["hr"].cumsum() - game_df["hr"]
    career_pa_sum = batter_group["pa"].cumsum() - game_df["pa"]
    game_df["career_hr_rate"] = np.where(
        career_pa_sum > 0,
        career_hr_sum / career_pa_sum,
        np.nan,
    )

    season_group = game_df.groupby(["batter", "season"])
    season_hr_sum = season_group["hr"].cumsum() - game_df["hr"]
    season_pa_sum = season_group["pa"].cumsum() - game_df["pa"]

    game_df["season_hr_rate"] = np.where(
        season_pa_sum > 0,
        season_hr_sum / season_pa_sum,
        np.nan,
    )

    return game_df


def build_pitcher_features(data: pd.DataFrame):
    pitcher_group_cols = ["game_date", "pitcher"]
    pitcher_split_group_cols = ["game_date", "pitcher", "stand"]

    sort_cols = ["pitcher", "game_date"]
    split_sort_cols = ["pitcher", "stand", "game_date"]
    if "game_pk" in data.columns:
        pitcher_group_cols.append("game_pk")
        pitcher_split_group_cols.append("game_pk")
        sort_cols.append("game_pk")
        split_sort_cols.append("game_pk")

    pitcher_df = (
        data.groupby(pitcher_group_cols)
        .agg(
            hr_allowed=("is_hr", "sum"),
            pa_allowed=("is_pa", "sum"),
        )
        .reset_index()
        .sort_values(sort_cols)
    )

    pitcher_df["hr_rate_allowed"] = np.where(
        pitcher_df["pa_allowed"] > 0,
        pitcher_df["hr_allowed"] / pitcher_df["pa_allowed"],
        np.nan,
    )

    pitcher_group = pitcher_df.groupby("pitcher")

    rolling_pitcher_hr_sum_30 = pitcher_group["hr_allowed"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).sum()
    )
    rolling_pitcher_pa_sum_30 = pitcher_group["pa_allowed"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).sum()
    )

    pitcher_df["rolling_pitcher_hr_30"] = np.where(
        rolling_pitcher_pa_sum_30 > 0,
        rolling_pitcher_hr_sum_30 / rolling_pitcher_pa_sum_30,
        np.nan,
    )

    pitcher_split_df = (
        data.groupby(pitcher_split_group_cols)
        .agg(
            hr_allowed=("is_hr", "sum"),
            pa_allowed=("is_pa", "sum"),
        )
        .reset_index()
        .sort_values(split_sort_cols)
    )

    split_group = pitcher_split_df.groupby(["pitcher", "stand"])

    rolling_split_hr_sum_30 = split_group["hr_allowed"].transform(
        lambda x: x.shift().rolling(30, min_periods=5).sum()
    )
    rolling_split_pa_sum_30 = split_group["pa_allowed"].transform(
        lambda x: x.shift().rolling(30, min_periods=5).sum()
    )

    pitcher_split_df["rolling_pitcher_hr_30_vs_batter_side"] = np.where(
        rolling_split_pa_sum_30 > 0,
        rolling_split_hr_sum_30 / rolling_split_pa_sum_30,
        np.nan,
    )

    pitcher_split_df = pitcher_split_df.rename(columns={"stand": "batter_side"})

    return pitcher_df, pitcher_split_df


def merge_pitcher_features(
    game_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
    pitcher_split_df: pd.DataFrame,
) -> pd.DataFrame:
    merge_keys = ["game_date", "pitcher"]
    split_left_keys = ["game_date", "pitcher", "stand"]
    split_right_keys = ["game_date", "pitcher", "batter_side"]

    if "game_pk" in game_df.columns and "game_pk" in pitcher_df.columns:
        merge_keys.append("game_pk")
    if "game_pk" in game_df.columns and "game_pk" in pitcher_split_df.columns:
        split_left_keys.insert(2, "game_pk")
        split_right_keys.insert(2, "game_pk")

    game_df = game_df.merge(
        pitcher_df[merge_keys + ["rolling_pitcher_hr_30"]],
        on=merge_keys,
        how="left",
    )

    game_df = game_df.merge(
        pitcher_split_df[
            split_right_keys + ["rolling_pitcher_hr_30_vs_batter_side"]
        ],
        left_on=split_left_keys,
        right_on=split_right_keys,
        how="left",
    )

    game_df = game_df.drop(columns=["batter_side"])

    return game_df


def add_matchup_features(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()

    game_df["is_same_hand_matchup"] = (
        game_df["stand"] == game_df["p_throws"]
    ).astype(int)

    game_df["is_platoon_matchup"] = (
        game_df["stand"] != game_df["p_throws"]
    ).astype(int)

    return game_df


def add_expected_pa_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected PA from recent batter usage only.
    Current-game lineup position is not available pregame in this pipeline,
    so do not derive PA expectations from realized same-game lineup order.
    """
    game_df = game_df.copy()
    sort_cols = ["game_date", "batter"]
    if "game_pk" in game_df.columns:
        sort_cols.append("game_pk")
    if "scheduled_game_no" in game_df.columns:
        sort_cols.append("scheduled_game_no")

    game_df = game_df.sort_values(sort_cols)

    batter_group = game_df.groupby("batter")

    game_df["rolling_pa_15"] = batter_group["pa"].transform(
        lambda x: x.shift().rolling(15, min_periods=5).mean()
    )

    game_df["rolling_pa_30"] = batter_group["pa"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    if "started" not in game_df.columns:
        game_df["started"] = np.nan

    game_df["rolling_start_rate_15"] = batter_group["started"].transform(
        lambda x: x.shift().rolling(15, min_periods=5).mean()
    )

    game_df["rolling_start_rate_30"] = batter_group["started"].transform(
        lambda x: x.shift().rolling(30, min_periods=10).mean()
    )

    game_df["expected_pa"] = np.where(
        game_df["rolling_pa_15"].notna() & game_df["rolling_pa_30"].notna(),
        0.65 * game_df["rolling_pa_15"] + 0.35 * game_df["rolling_pa_30"],
        game_df["rolling_pa_15"]
    )

    game_df["expected_pa"] = game_df["expected_pa"].fillna(game_df["rolling_pa_30"])

    game_df["start_rate_recent"] = np.where(
        game_df["rolling_start_rate_15"].notna() & game_df["rolling_start_rate_30"].notna(),
        0.65 * game_df["rolling_start_rate_15"] + 0.35 * game_df["rolling_start_rate_30"],
        game_df["rolling_start_rate_15"]
    )
    game_df["start_rate_recent"] = game_df["start_rate_recent"].fillna(game_df["rolling_start_rate_30"])

    game_df["_feature_row_id"] = np.arange(len(game_df))
    matchup_specs = [("R", "rhp"), ("L", "lhp")]

    for hand, suffix in matchup_specs:
        hand_df = game_df.loc[
            game_df["p_throws"] == hand,
            ["_feature_row_id", "batter", "pa", "started"],
        ].copy()

        if hand_df.empty:
            game_df[f"expected_pa_vs_{suffix}"] = np.nan
            game_df[f"start_rate_vs_{suffix}"] = np.nan
            continue

        hand_group = hand_df.groupby("batter")
        hand_df[f"rolling_pa_15_vs_{suffix}"] = hand_group["pa"].transform(
            lambda x: x.shift().rolling(15, min_periods=3).mean()
        )
        hand_df[f"rolling_pa_30_vs_{suffix}"] = hand_group["pa"].transform(
            lambda x: x.shift().rolling(30, min_periods=5).mean()
        )
        hand_df[f"rolling_start_rate_15_vs_{suffix}"] = hand_group["started"].transform(
            lambda x: x.shift().rolling(15, min_periods=3).mean()
        )
        hand_df[f"rolling_start_rate_30_vs_{suffix}"] = hand_group["started"].transform(
            lambda x: x.shift().rolling(30, min_periods=5).mean()
        )

        hand_df[f"expected_pa_vs_{suffix}"] = np.where(
            hand_df[f"rolling_pa_15_vs_{suffix}"].notna() &
            hand_df[f"rolling_pa_30_vs_{suffix}"].notna(),
            0.65 * hand_df[f"rolling_pa_15_vs_{suffix}"] +
            0.35 * hand_df[f"rolling_pa_30_vs_{suffix}"],
            hand_df[f"rolling_pa_15_vs_{suffix}"],
        )
        hand_df[f"expected_pa_vs_{suffix}"] = hand_df[f"expected_pa_vs_{suffix}"].fillna(
            hand_df[f"rolling_pa_30_vs_{suffix}"]
        )

        hand_df[f"start_rate_vs_{suffix}"] = np.where(
            hand_df[f"rolling_start_rate_15_vs_{suffix}"].notna() &
            hand_df[f"rolling_start_rate_30_vs_{suffix}"].notna(),
            0.65 * hand_df[f"rolling_start_rate_15_vs_{suffix}"] +
            0.35 * hand_df[f"rolling_start_rate_30_vs_{suffix}"],
            hand_df[f"rolling_start_rate_15_vs_{suffix}"],
        )
        hand_df[f"start_rate_vs_{suffix}"] = hand_df[f"start_rate_vs_{suffix}"].fillna(
            hand_df[f"rolling_start_rate_30_vs_{suffix}"]
        )

        game_df = game_df.merge(
            hand_df[["_feature_row_id", f"expected_pa_vs_{suffix}", f"start_rate_vs_{suffix}"]],
            on="_feature_row_id",
            how="left",
        )

    for suffix in ["rhp", "lhp"]:
        game_df[f"expected_pa_vs_{suffix}"] = (
            game_df.groupby("batter")[f"expected_pa_vs_{suffix}"].ffill()
        )
        game_df[f"start_rate_vs_{suffix}"] = (
            game_df.groupby("batter")[f"start_rate_vs_{suffix}"].ffill()
        )

    game_df["expected_pa_vs_rhp"] = game_df["expected_pa_vs_rhp"].fillna(game_df["expected_pa"])
    game_df["expected_pa_vs_lhp"] = game_df["expected_pa_vs_lhp"].fillna(game_df["expected_pa"])
    game_df["start_rate_vs_rhp"] = game_df["start_rate_vs_rhp"].fillna(game_df["start_rate_recent"])
    game_df["start_rate_vs_lhp"] = game_df["start_rate_vs_lhp"].fillna(game_df["start_rate_recent"])

    game_df["expected_pa_matchup"] = np.where(
        game_df["p_throws"] == "R",
        game_df["expected_pa_vs_rhp"],
        game_df["expected_pa_vs_lhp"],
    )
    game_df["start_rate_matchup"] = np.where(
        game_df["p_throws"] == "R",
        game_df["start_rate_vs_rhp"],
        game_df["start_rate_vs_lhp"],
    )
    game_df["platoon_start_rate_gap"] = (
        game_df["start_rate_vs_lhp"] - game_df["start_rate_vs_rhp"]
    )

    game_df = game_df.drop(columns=["_feature_row_id"])

    return game_df


def add_lineup_features(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()

    if "lineup_position" not in game_df.columns:
        game_df["lineup_position"] = np.nan

    game_df["lineup_position"] = pd.to_numeric(game_df["lineup_position"], errors="coerce")
    game_df["lineup_position_filled"] = game_df["lineup_position"].fillna(9.5)
    game_df["lineup_top_third"] = game_df["lineup_position"].between(1, 3, inclusive="both").fillna(False).astype(int)
    game_df["lineup_middle_third"] = game_df["lineup_position"].between(4, 6, inclusive="both").fillna(False).astype(int)
    game_df["lineup_bottom_third"] = game_df["lineup_position"].between(7, 9, inclusive="both").fillna(False).astype(int)

    # Smooth mapping from lineup slot to expected opportunity once lineups are known.
    lineup_pa_map = {
        1: 4.75,
        2: 4.60,
        3: 4.50,
        4: 4.40,
        5: 4.25,
        6: 4.10,
        7: 3.95,
        8: 3.80,
        9: 3.65,
    }
    game_df["lineup_expected_pa"] = game_df["lineup_position"].map(lineup_pa_map)
    game_df["lineup_expected_pa"] = game_df["lineup_expected_pa"].fillna(game_df["expected_pa"])

    return game_df


def add_weather_features(
    game_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    game_df = game_df.copy()
    weather_df = weather_df.copy()

    # Make both merge keys naive midnight timestamps
    game_df["game_date"] = pd.to_datetime(game_df["game_date"]).dt.tz_localize(None).dt.normalize()
    weather_df["game_date"] = pd.to_datetime(weather_df["game_date"]).dt.tz_localize(None).dt.normalize()

    merge_keys = ["game_date", "home_team"]

    if "away_team" in weather_df.columns:
        merge_keys.append("away_team")

    if "scheduled_game_no" in game_df.columns and "scheduled_game_no" in weather_df.columns:
        merge_keys.append("scheduled_game_no")

    weather_cols = merge_keys + [
        col for col in weather_df.columns
        if col not in merge_keys and col != "game_id"
    ]

    game_df = game_df.merge(
        weather_df[weather_cols].drop_duplicates(),
        on=merge_keys,
        how="left"
    )

    game_df["weather_missing_raw"] = game_df["temperature_f"].isna().astype(int)
    game_df["weather_month"] = pd.to_datetime(game_df["game_date"]).dt.month

    for col in ["temperature_f", "humidity", "wind_speed_mph"]:
        team_month_median = (
            game_df.groupby(["home_team", "weather_month"])[col]
            .transform("median")
        )
        team_median = game_df.groupby("home_team")[col].transform("median")
        global_median = game_df[col].median()
        game_df[col] = game_df[col].fillna(team_month_median).fillna(team_median).fillna(global_median)

    if "roof_closed" not in game_df.columns:
        game_df["roof_closed"] = 0

    if "humidity" not in game_df.columns:
        game_df["humidity"] = np.nan

    game_df["roof_closed"] = game_df["roof_closed"].fillna(0).astype(int)
    direction = game_df["wind_direction"].fillna("unknown").astype(str).str.lower()

    game_df["wind_out"] = direction.isin(
        ["out", "out to cf", "out to center", "out to left", "out to right"]
    ).astype(int)

    game_df["wind_in"] = direction.isin(
        ["in", "in from cf", "in from center", "in from left", "in from right"]
    ).astype(int)

    game_df["crosswind"] = direction.str.contains("cross").fillna(False).astype(int)

    game_df["temperature_x_wind"] = game_df["temperature_f"] * game_df["wind_speed_mph"]
    game_df["wind_out_effect"] = game_df["wind_speed_mph"] * game_df["wind_out"]
    game_df["wind_in_effect"] = game_df["wind_speed_mph"] * game_df["wind_in"]
    game_df["temperature_sq"] = game_df["temperature_f"] ** 2

    game_df.loc[
        game_df["roof_closed"] == 1,
        ["wind_out_effect", "wind_in_effect"]
    ] = 0

    game_df = game_df.drop(columns=["weather_month"])

    return game_df


def add_handedness_park_factors(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()

    park_hr_factor_by_hand = {
        "ARI": {"L": 0.77, "R": 0.97},
        "ATH": {"L": 1.17, "R": 1.08},
        "ATL": {"L": 1.09, "R": 1.01},
        "BAL": {"L": 1.25, "R": 0.87},
        "BOS": {"L": 0.87, "R": 0.90},
        "CHC": {"L": 0.95, "R": 1.02},
        "CWS": {"L": 0.96, "R": 0.93},
        "CIN": {"L": 1.26, "R": 1.21},
        "CLE": {"L": 0.93, "R": 0.75},
        "COL": {"L": 1.05, "R": 1.06},
        "DET": {"L": 0.99, "R": 1.00},
        "HOU": {"L": 1.10, "R": 1.03},
        "KC": {"L": 0.73, "R": 0.93},
        "LAA": {"L": 1.08, "R": 1.16},
        "LAD": {"L": 1.19, "R": 1.35},
        "MIA": {"L": 0.98, "R": 0.84},
        "MIL": {"L": 1.00, "R": 1.10},
        "MIN": {"L": 1.02, "R": 1.02},
        "NYM": {"L": 0.98, "R": 1.09},
        "NYY": {"L": 1.18, "R": 1.19},
        "OAK": {"L": 1.17, "R": 1.08},
        "PHI": {"L": 1.28, "R": 1.03},
        "PIT": {"L": 0.87, "R": 0.68},
        "SD": {"L": 0.90, "R": 1.13},
        "SEA": {"L": 0.95, "R": 0.91},
        "SF": {"L": 0.78, "R": 0.84},
        "STL": {"L": 0.89, "R": 0.85},
        "TB": {"L": 0.85, "R": 1.11},
        "TEX": {"L": 1.02, "R": 1.05},
        "TOR": {"L": 1.02, "R": 1.05},
        "WSH": {"L": 0.97, "R": 0.91},
    }

    def lookup_pf(team, stand):
        team_dict = park_hr_factor_by_hand.get(team, None)
        if team_dict is None:
            return 1.0
        return team_dict.get(stand, 1.0)

    game_df["park_hr_factor_by_hand"] = game_df.apply(
        lambda row: lookup_pf(normalize_team_abbr(row["home_team"], row["game_date"]), row["stand"]),
        axis=1
    )

    return game_df

def add_interaction_features(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()

    game_df["hr_power_form"] = (
        game_df["rolling_hr_rate_30"] *
        game_df["expected_pa"]
    )

    game_df["park_power"] = (
        game_df["park_hr_factor_by_hand"] *
        game_df["rolling_flyball_rate_30"]
    )

    game_df["matchup_power"] = (
        game_df["rolling_hr_rate_30"] *
        game_df["rolling_pitcher_hr_30_vs_batter_side"]
    )

    if "temperature_f" in game_df.columns:
        game_df["barrel_env"] = (
            game_df["rolling_barrel_30"] *
            game_df["temperature_f"]
        )
    else:
        game_df["barrel_env"] = game_df["rolling_barrel_30"]

    return game_df


def build_feature_dataset(
    data: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = add_contact_flags(data)
    data = add_lineup_position(data)

    game_df = build_game_dataset(data)
    game_df = add_scheduled_game_number(game_df)
    game_df = add_hitter_rolling_features(game_df)

    pitcher_df, pitcher_split_df = build_pitcher_features(data)
    game_df = merge_pitcher_features(game_df, pitcher_df, pitcher_split_df)

    game_df = add_matchup_features(game_df)
    game_df = add_expected_pa_features(game_df)
    game_df = add_lineup_features(game_df)
    game_df = add_handedness_park_factors(game_df)

    if weather_df is not None:
        game_df = add_weather_features(game_df, weather_df)

    game_df = add_interaction_features(game_df)
    game_df = add_batter_names(game_df)

    return game_df


def get_model_features(
    include_weather: bool = True,
    include_platoon_pa: bool = False,
    include_lineup_position: bool = False,
):
    features = [
        "season",
        "expected_pa",
        "rolling_pa_15",
        "rolling_pa_30",
        "rolling_hr_15",
        "rolling_hr_30",
        "rolling_hr_rate_30",
        "rolling_barrel_30",
        "rolling_hardhit_30",
        "rolling_ev_30",
        "rolling_max_ev_30",
        "rolling_pull_rate_30",
        "rolling_flyball_rate_30",
        "rolling_xslg_30",
        "rolling_slg_30",
        "rolling_avg_30",
        "rolling_iso_30",
        "career_hr_rate",
        "season_hr_rate",
        "rolling_pitcher_hr_30",
        "rolling_pitcher_hr_30_vs_batter_side",
        "is_same_hand_matchup",
        "is_platoon_matchup",
        "park_hr_factor_by_hand",
        "hr_power_form",
        "park_power",
        "matchup_power",
        "barrel_env",
    ]

    if include_platoon_pa:
        features += [
            "expected_pa_vs_rhp",
            "expected_pa_vs_lhp",
            "expected_pa_matchup",
            "start_rate_vs_rhp",
            "start_rate_vs_lhp",
            "start_rate_matchup",
            "platoon_start_rate_gap",
        ]

    if include_lineup_position:
        features += [
            "lineup_position_filled",
            "lineup_top_third",
            "lineup_middle_third",
            "lineup_bottom_third",
            "lineup_expected_pa",
        ]

    if include_weather:
        features += [
            "temperature_f",
            "wind_speed_mph",
            "humidity",
            "roof_closed",
            "weather_missing_raw",
            "wind_out",
            "wind_in",
            "crosswind",
            "temperature_x_wind",
            "wind_out_effect",
            "wind_in_effect",
            "temperature_sq",
        ]

    return features

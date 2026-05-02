import pickle
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pybaseball import cache, statcast

from feature_engineering import (
    add_contact_flags,
    add_handedness_park_factors,
    add_interaction_features,
    add_matchup_features,
    add_weather_features,
    add_batter_names,
    build_feature_dataset,
    build_pitcher_features,
)
from odds_features import (
    add_scheduled_game_number,
    add_edge_features,
    american_to_prob,
    apply_name_alias,
    normalize_team_name_to_abbr,
    normalize_player_name,
    prepare_odds_features,
)
from probable_pitchers import build_matchup_view, fetch_probable_pitchers
from pullLiveOdds import fetch_live_hr_odds
from live_weather import build_live_weather_df


try:
    cache.enable()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_BUNDLE_PATH = BASE_DIR / "saved_models/residual_late_snapshot_champion.pkl"
HISTORICAL_WEATHER_PATH = BASE_DIR / "weather_from_odds.parquet"
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_STATCAST_START_YEAR = 2023
DEFAULT_BOOKMAKERS = "draftkings,fanduel,betmgm,caesars,betrivers,espnbet,novig,prophetx"
ROSTER_TYPE_PRIORITY = {
    "gameday": 0,
    "active": 1,
    "depthChart": 2,
    "40Man": 3,
    "fullRoster": 4,
    "fullSeason": 5,
    "nonRosterInvitees": 6,
}
LIVE_FEATURE_CACHE_DIR = BASE_DIR / "live_feature_cache"
LIVE_BATTER_FEATURES_PATH = LIVE_FEATURE_CACHE_DIR / "latest_batter_features.parquet"
LIVE_PITCHER_FEATURES_PATH = LIVE_FEATURE_CACHE_DIR / "latest_pitcher_features.parquet"
LIVE_PITCHER_SPLIT_FEATURES_PATH = LIVE_FEATURE_CACHE_DIR / "latest_pitcher_split_features.parquet"
LIVE_FEATURE_METADATA_PATH = LIVE_FEATURE_CACHE_DIR / "metadata.json"
STATCAST_HISTORY_CACHE_PATH = LIVE_FEATURE_CACHE_DIR / "statcast_history.parquet"


def load_model_bundle(path: Path = MODEL_BUNDLE_PATH) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def current_mlb_season_start(target_date: date) -> date:
    return date(target_date.year, 3, 1)


def build_statcast_ranges_between(
    start_date: date,
    end_date: date,
) -> list[tuple[str, str]]:
    if end_date < start_date:
        return []

    ranges = []
    for year in range(start_date.year, end_date.year + 1):
        season_start = date(year, 3, 1)
        season_end = date(year, 11, 15)
        start = max(start_date, season_start)
        end = min(end_date, season_end)
        if start <= end:
            ranges.append((start.isoformat(), end.isoformat()))
    return ranges


def build_statcast_ranges(
    target_date: date,
    start_year: int = DEFAULT_STATCAST_START_YEAR,
) -> list[tuple[str, str]]:
    end_date = target_date - timedelta(days=1)
    if end_date < date(start_year, 1, 1):
        return []
    return build_statcast_ranges_between(
        start_date=max(date(start_year, 1, 1), date(start_year, 3, 1)),
        end_date=end_date,
    )


def load_statcast_history_for_live(
    target_date: date,
    start_year: int = DEFAULT_STATCAST_START_YEAR,
) -> pd.DataFrame:
    ranges = build_statcast_ranges(target_date=target_date, start_year=start_year)
    if not ranges:
        return pd.DataFrame()

    parts = []
    for start_dt, end_dt in ranges:
        part = statcast(start_dt=start_dt, end_dt=end_dt)
        if part is not None and not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame()

    data = pd.concat(parts, ignore_index=True)
    data["game_date"] = pd.to_datetime(data["game_date"])
    return data


def _normalize_statcast_history_df(statcast_df: pd.DataFrame) -> pd.DataFrame:
    if statcast_df is None or statcast_df.empty:
        return pd.DataFrame()
    normalized = statcast_df.copy()
    normalized["game_date"] = pd.to_datetime(normalized["game_date"])
    subset = [col for col in ["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"] if col in normalized.columns]
    if subset:
        normalized = normalized.drop_duplicates(subset=subset, keep="last")
    else:
        normalized = normalized.drop_duplicates()
    return normalized.sort_values([col for col in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if col in normalized.columns]).reset_index(drop=True)


def load_cached_statcast_history(cache_path: Path = STATCAST_HISTORY_CACHE_PATH) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    return _normalize_statcast_history_df(pd.read_parquet(cache_path))


def save_cached_statcast_history(statcast_df: pd.DataFrame, cache_path: Path = STATCAST_HISTORY_CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_statcast_history_df(statcast_df)
    normalized.to_parquet(cache_path, index=False)


def load_incremental_statcast_history_for_live(
    target_date: date,
    start_year: int = DEFAULT_STATCAST_START_YEAR,
    cache_path: Path = STATCAST_HISTORY_CACHE_PATH,
    force_full_refresh: bool = False,
) -> pd.DataFrame:
    end_date = target_date - timedelta(days=1)
    if end_date < date(start_year, 1, 1):
        return pd.DataFrame()

    cached_df = pd.DataFrame() if force_full_refresh else load_cached_statcast_history(cache_path)
    if cached_df.empty:
        fresh_df = load_statcast_history_for_live(target_date=target_date, start_year=start_year)
        save_cached_statcast_history(fresh_df, cache_path=cache_path)
        return fresh_df

    cached_start = pd.Timestamp(cached_df["game_date"].min()).date()
    # Early March often has no Statcast rows yet, so only treat the cache as incompatible
    # when it starts in a later season than the requested start year.
    if cached_start.year > start_year:
        fresh_df = load_statcast_history_for_live(target_date=target_date, start_year=start_year)
        save_cached_statcast_history(fresh_df, cache_path=cache_path)
        return fresh_df

    cached_end = pd.Timestamp(cached_df["game_date"].max()).date()
    if cached_end >= end_date:
        return cached_df

    ranges = build_statcast_ranges_between(start_date=cached_end + timedelta(days=1), end_date=end_date)
    if not ranges:
        return cached_df

    parts = [cached_df]
    for start_dt, end_dt in ranges:
        part = statcast(start_dt=start_dt, end_dt=end_dt)
        if part is not None and not part.empty:
            parts.append(part)

    combined = _normalize_statcast_history_df(pd.concat(parts, ignore_index=True))
    save_cached_statcast_history(combined, cache_path=cache_path)
    return combined


def build_historical_feature_tables(
    statcast_df: pd.DataFrame,
    historical_weather_path: Path = HISTORICAL_WEATHER_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if statcast_df.empty:
        raise ValueError("Statcast history is empty.")

    weather_df = pd.read_parquet(historical_weather_path) if historical_weather_path.exists() else None
    game_df = build_feature_dataset(statcast_df, weather_df=weather_df)

    working = add_contact_flags(statcast_df.copy())
    pitcher_df, pitcher_split_df = build_pitcher_features(working)
    return game_df, pitcher_df, pitcher_split_df


def _latest_by_group(df: pd.DataFrame, group_cols: list[str], sort_cols: list[str]) -> pd.DataFrame:
    return (
        df.sort_values(sort_cols)
        .groupby(group_cols, as_index=False, group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )


def build_latest_batter_feature_table(game_df: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_by_group(
        game_df,
        group_cols=["batter"],
        sort_cols=["game_date", "scheduled_game_no", "batter"],
    ).copy()
    latest["player_name_norm"] = (
        latest["player_name"]
        .apply(normalize_player_name)
        .apply(apply_name_alias)
    )
    return latest


def build_latest_pitcher_feature_tables(
    pitcher_df: pd.DataFrame,
    pitcher_split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_pitchers = _latest_by_group(
        pitcher_df,
        group_cols=["pitcher"],
        sort_cols=["game_date", "pitcher"],
    )[["pitcher", "rolling_pitcher_hr_30"]].copy()

    latest_pitcher_splits = _latest_by_group(
        pitcher_split_df,
        group_cols=["pitcher", "batter_side"],
        sort_cols=["game_date", "pitcher", "batter_side"],
    )[["pitcher", "batter_side", "rolling_pitcher_hr_30_vs_batter_side"]].copy()

    return latest_pitchers, latest_pitcher_splits


def save_precomputed_live_feature_cache(
    latest_batter_df: pd.DataFrame,
    latest_pitcher_df: pd.DataFrame,
    latest_pitcher_split_df: pd.DataFrame,
    target_date: date,
    cache_dir: Path = LIVE_FEATURE_CACHE_DIR,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    batter_path = cache_dir / LIVE_BATTER_FEATURES_PATH.name
    pitcher_path = cache_dir / LIVE_PITCHER_FEATURES_PATH.name
    pitcher_split_path = cache_dir / LIVE_PITCHER_SPLIT_FEATURES_PATH.name
    metadata_path = cache_dir / LIVE_FEATURE_METADATA_PATH.name

    latest_batter_df.to_parquet(batter_path, index=False)
    latest_pitcher_df.to_parquet(pitcher_path, index=False)
    latest_pitcher_split_df.to_parquet(pitcher_split_path, index=False)

    statcast_history_path = cache_dir / STATCAST_HISTORY_CACHE_PATH.name
    statcast_history_rows = None
    statcast_history_through = None
    if statcast_history_path.exists():
        try:
            statcast_history_df = pd.read_parquet(statcast_history_path, columns=["game_date"])
            statcast_history_rows = int(len(statcast_history_df))
            if not statcast_history_df.empty:
                statcast_history_through = pd.to_datetime(statcast_history_df["game_date"]).max().date().isoformat()
        except Exception:
            statcast_history_rows = None
            statcast_history_through = None

    metadata = {
        "built_for_target_date": target_date.isoformat(),
        "built_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "latest_batter_rows": int(len(latest_batter_df)),
        "latest_pitcher_rows": int(len(latest_pitcher_df)),
        "latest_pitcher_split_rows": int(len(latest_pitcher_split_df)),
        "statcast_history_rows": statcast_history_rows,
        "statcast_history_through": statcast_history_through,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def load_precomputed_live_feature_cache(
    cache_dir: Path = LIVE_FEATURE_CACHE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    batter_path = cache_dir / LIVE_BATTER_FEATURES_PATH.name
    pitcher_path = cache_dir / LIVE_PITCHER_FEATURES_PATH.name
    pitcher_split_path = cache_dir / LIVE_PITCHER_SPLIT_FEATURES_PATH.name
    metadata_path = cache_dir / LIVE_FEATURE_METADATA_PATH.name
    required_paths = [
        batter_path,
        pitcher_path,
        pitcher_split_path,
        metadata_path,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing live feature cache files: "
            + ", ".join(path.name for path in missing)
        )

    latest_batter_df = pd.read_parquet(batter_path)
    latest_pitcher_df = pd.read_parquet(pitcher_path)
    latest_pitcher_split_df = pd.read_parquet(pitcher_split_path)
    metadata = json.loads(metadata_path.read_text())
    return latest_batter_df, latest_pitcher_df, latest_pitcher_split_df, metadata


def build_and_save_live_feature_cache(
    target_date: date,
    start_year: int = DEFAULT_STATCAST_START_YEAR,
    historical_weather_path: Path = HISTORICAL_WEATHER_PATH,
    cache_dir: Path = LIVE_FEATURE_CACHE_DIR,
    force_full_refresh: bool = False,
) -> dict:
    statcast_df = load_incremental_statcast_history_for_live(
        target_date=target_date,
        start_year=start_year,
        cache_path=cache_dir / STATCAST_HISTORY_CACHE_PATH.name,
        force_full_refresh=force_full_refresh,
    )
    game_df, pitcher_df, pitcher_split_df = build_historical_feature_tables(
        statcast_df=statcast_df,
        historical_weather_path=historical_weather_path,
    )
    latest_batter_df = build_latest_batter_feature_table(game_df)
    latest_pitcher_df, latest_pitcher_split_df = build_latest_pitcher_feature_tables(
        pitcher_df,
        pitcher_split_df,
    )
    return save_precomputed_live_feature_cache(
        latest_batter_df=latest_batter_df,
        latest_pitcher_df=latest_pitcher_df,
        latest_pitcher_split_df=latest_pitcher_split_df,
        target_date=target_date,
        cache_dir=cache_dir,
    )


def fetch_team_roster(
    team_id: int,
    target_date: date,
    roster_type: str,
    timeout: int = 30,
) -> pd.DataFrame:
    url = f"{MLB_STATS_API_BASE}/teams/{team_id}/roster"
    response = requests.get(
        url,
        params={"rosterType": roster_type, "date": target_date.isoformat()},
        timeout=timeout,
    )
    if not response.ok:
        return pd.DataFrame()
    payload = response.json()

    rows = []
    for row in payload.get("roster", []):
        person = row.get("person", {}) or {}
        rows.append(
            {
                "player_id": person.get("id"),
                "player_name": person.get("fullName"),
                "player_name_norm": apply_name_alias(normalize_player_name(person.get("fullName"))),
                "roster_type": roster_type,
            }
        )
    return pd.DataFrame(rows)


def build_game_roster_map(matchups_df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    roster_rows = []
    for row in matchups_df.itertuples():
        side_specs = [
            ("away", row.away_team_id, row.away_team),
            ("home", row.home_team_id, row.home_team),
        ]
        for team_side, team_id, team_abbr in side_specs:
            if pd.isna(team_id):
                continue
            for roster_type in ["gameday", "active", "depthChart", "40Man", "fullRoster", "fullSeason", "nonRosterInvitees"]:
                roster_df = fetch_team_roster(int(team_id), target_date=target_date, roster_type=roster_type)
                if roster_df.empty:
                    continue
                roster_df["game_date"] = pd.to_datetime(row.game_date).normalize()
                roster_df["home_team"] = row.home_team
                roster_df["away_team"] = row.away_team
                roster_df["scheduled_game_no"] = row.scheduled_game_no
                roster_df["batting_team"] = team_abbr
                roster_df["team_side"] = team_side
                roster_df["roster_priority"] = roster_df["roster_type"].map(ROSTER_TYPE_PRIORITY).fillna(999)
                roster_rows.append(roster_df)

    if not roster_rows:
        return pd.DataFrame()

    roster_map = pd.concat(roster_rows, ignore_index=True)
    roster_map = roster_map.sort_values(
        ["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm", "roster_priority", "roster_type"]
    )
    roster_map = roster_map.drop_duplicates(
        subset=["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm"],
        keep="first",
    ).reset_index(drop=True)
    return roster_map


def build_live_matchups(
    target_date: date,
    game_type: str | None = None,
) -> pd.DataFrame:
    probables_df = fetch_probable_pitchers(
        start_date=target_date.isoformat(),
        game_type=game_type,
    )
    matchups_df = build_matchup_view(probables_df)
    if matchups_df.empty:
        return matchups_df

    matchups_df["game_date"] = pd.to_datetime(matchups_df["game_date"]).dt.normalize()
    matchups_df = matchups_df.sort_values(
        ["game_date", "home_team", "away_team", "game_time_utc", "game_pk"]
    ).reset_index(drop=True)
    matchups_df["scheduled_game_no"] = (
        matchups_df.groupby(["game_date", "home_team", "away_team"]).cumcount() + 1
    )
    return matchups_df


def build_live_input_frames(
    target_date: date,
    bookmakers: str | None = None,
    game_type: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    odds_raw_df = fetch_live_hr_odds(
        bookmakers=bookmakers,
        snapshot_label="live",
    )
    odds_features_df = prepare_odds_features(odds_raw_df)
    matchups_df = build_live_matchups(target_date=target_date, game_type=game_type)
    weather_df = build_live_weather_df(start_date=target_date.isoformat(), game_type=game_type)
    roster_map_df = build_game_roster_map(matchups_df, target_date=target_date) if not matchups_df.empty else pd.DataFrame()
    return odds_raw_df, odds_features_df, matchups_df, weather_df, roster_map_df


def build_live_candidate_rows(
    target_date: date,
    latest_batter_df: pd.DataFrame,
    latest_pitcher_df: pd.DataFrame,
    latest_pitcher_split_df: pd.DataFrame,
    odds_features_df: pd.DataFrame,
    matchups_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    roster_map_df: pd.DataFrame,
) -> pd.DataFrame:
    if odds_features_df.empty:
        return pd.DataFrame()

    candidates = odds_features_df.copy()
    candidates["game_date"] = pd.to_datetime(candidates["game_date"]).dt.normalize()

    if not roster_map_df.empty:
        candidates = candidates.merge(
            roster_map_df[[
                "game_date",
                "home_team",
                "away_team",
                "scheduled_game_no",
                "player_name_norm",
                "batting_team",
                "team_side",
            ]].drop_duplicates(),
            on=["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm"],
            how="left",
        )

    latest_batter_merge = latest_batter_df.rename(
        columns={
            col: f"hist_{col}" for col in latest_batter_df.columns if col != "player_name_norm"
        }
    )

    candidates = candidates.merge(
        latest_batter_merge,
        on="player_name_norm",
        how="left",
    )

    historical_passthrough_cols = [
        col for col in latest_batter_df.columns
        if col not in {"player_name_norm", "game_date", "home_team", "away_team", "scheduled_game_no", "game_pk"}
    ]
    for col in historical_passthrough_cols:
        hist_col = f"hist_{col}"
        if hist_col not in candidates.columns:
            continue
        if col in candidates.columns:
            candidates[col] = candidates[col].fillna(candidates[hist_col])
        else:
            candidates[col] = candidates[hist_col]

    if "player_name" not in candidates.columns:
        candidates["player_name"] = np.nan
    if "player" in candidates.columns:
        candidates["player_name"] = candidates["player_name"].fillna(candidates["player"])
    candidates["player_name"] = candidates["player_name"]
    candidates["batter"] = candidates["batter"]
    candidates["stand"] = candidates["stand"]
    candidates["started"] = candidates["started"]
    candidates["season"] = target_date.year

    candidates = candidates.merge(
        matchups_df[[
            "game_date",
            "home_team",
            "away_team",
            "scheduled_game_no",
            "game_pk",
            "game_time_utc",
            "away_pitcher_id",
            "away_probable_pitcher",
            "away_pitcher_throws",
            "home_pitcher_id",
            "home_probable_pitcher",
            "home_pitcher_throws",
        ]],
        on=["game_date", "home_team", "away_team", "scheduled_game_no"],
        how="left",
    )

    # Drop stale historical opponent-pitcher fields from the cached batter
    # snapshot before constructing today's probable-pitcher matchup columns.
    stale_pitcher_cols = [
        "pitcher",
        "p_throws",
        "rolling_pitcher_hr_30",
        "rolling_pitcher_hr_30_vs_batter_side",
        "is_same_hand_matchup",
        "is_platoon_matchup",
        "matchup_power",
    ]
    existing_stale_pitcher_cols = [col for col in stale_pitcher_cols if col in candidates.columns]
    if existing_stale_pitcher_cols:
        candidates = candidates.drop(columns=existing_stale_pitcher_cols)

    if "batting_team" not in candidates.columns:
        candidates["batting_team"] = np.nan
    candidates["batting_team"] = candidates["batting_team"].fillna(candidates["hist_batting_team"])
    candidates["pitcher"] = np.where(
        candidates["batting_team"] == candidates["away_team"],
        candidates["home_pitcher_id"],
        np.where(
            candidates["batting_team"] == candidates["home_team"],
            candidates["away_pitcher_id"],
            np.nan,
        ),
    )
    candidates["opposing_pitcher_name"] = np.where(
        candidates["batting_team"] == candidates["away_team"],
        candidates["home_probable_pitcher"],
        np.where(
            candidates["batting_team"] == candidates["home_team"],
            candidates["away_probable_pitcher"],
            None,
        ),
    )
    candidates["p_throws"] = np.where(
        candidates["batting_team"] == candidates["away_team"],
        candidates["home_pitcher_throws"],
        np.where(
            candidates["batting_team"] == candidates["home_team"],
            candidates["away_pitcher_throws"],
            None,
        ),
    )

    candidates = candidates.merge(
        latest_pitcher_df.rename(columns={"pitcher": "pitcher_lookup"}),
        left_on="pitcher",
        right_on="pitcher_lookup",
        how="left",
    ).drop(columns=["pitcher_lookup"])

    candidates = candidates.merge(
        latest_pitcher_split_df.rename(columns={"pitcher": "pitcher_lookup"}),
        left_on=["pitcher", "stand"],
        right_on=["pitcher_lookup", "batter_side"],
        how="left",
    ).drop(columns=["pitcher_lookup", "batter_side"])

    candidates["game_date"] = pd.to_datetime(candidates["game_date"]).dt.normalize()
    candidates["started"] = candidates["started"].fillna(1)
    candidates = add_matchup_features(candidates)
    candidates = add_handedness_park_factors(candidates)

    # Drop historical weather columns from the cached batter snapshot so the
    # live weather merge can write the canonical field names without suffixes.
    stale_weather_cols = [
        "temperature_f",
        "wind_speed_mph",
        "humidity",
        "roof_closed",
        "weather_missing_raw",
        "wind_direction",
        "wind_out",
        "wind_in",
        "crosswind",
        "temperature_x_wind",
        "wind_out_effect",
        "wind_in_effect",
        "temperature_sq",
    ]
    existing_stale_weather_cols = [col for col in stale_weather_cols if col in candidates.columns]
    if existing_stale_weather_cols:
        candidates = candidates.drop(columns=existing_stale_weather_cols)

    if not weather_df.empty:
        candidates = add_weather_features(candidates, weather_df)
    else:
        for col in ["temperature_f", "wind_speed_mph", "humidity", "roof_closed", "wind_direction"]:
            if col not in candidates.columns:
                candidates[col] = np.nan
        candidates["weather_missing_raw"] = 1
        candidates["wind_out"] = 0
        candidates["wind_in"] = 0
        candidates["crosswind"] = 0
        candidates["temperature_x_wind"] = np.nan
        candidates["wind_out_effect"] = 0.0
        candidates["wind_in_effect"] = 0.0
        candidates["temperature_sq"] = np.nan

    candidates = add_interaction_features(candidates)

    return candidates


def score_live_candidates(candidates_df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    if candidates_df.empty:
        return pd.DataFrame()

    scored = candidates_df.copy()
    feature_cols = model_bundle["features"]
    market_prob_col = model_bundle["market_prob_col"]

    raw_probs = np.clip(
        scored[market_prob_col].to_numpy() + model_bundle["model"].predict(scored[feature_cols]),
        0,
        1,
    )
    scored["raw_model_prob"] = raw_probs
    scored["model_prob"] = model_bundle["calibrator"].predict_proba(
        pd.DataFrame({"raw_prob": raw_probs})
    )[:, 1]
    scored = add_edge_features(scored, model_prob_col="model_prob")
    return scored


def american_to_profit(odds: float | int | None) -> float | None:
    if odds is None or pd.isna(odds):
        return np.nan
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)


def profit_to_american(profit_multiple: float | int | None) -> float | None:
    if profit_multiple is None or pd.isna(profit_multiple) or profit_multiple <= 0:
        return np.nan
    if profit_multiple >= 1:
        return round(profit_multiple * 100)
    return round(-100 / profit_multiple)


def apply_profit_boost_to_odds(odds: float | int | None, boost_pct: float) -> float | None:
    if odds is None or pd.isna(odds):
        return np.nan
    profit_multiple = american_to_profit(odds)
    boosted_profit = profit_multiple * (1 + boost_pct / 100.0)
    return profit_to_american(boosted_profit)


def fill_pitcher_context_from_batting_team(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "batting_team" not in working.columns:
        working["batting_team"] = np.nan
    if "opposing_pitcher_name" not in working.columns:
        working["opposing_pitcher_name"] = np.nan
    if "p_throws" not in working.columns:
        working["p_throws"] = np.nan

    missing_pitcher = working["opposing_pitcher_name"].isna()
    away_batting = working["batting_team"] == working["away_team"]
    home_batting = working["batting_team"] == working["home_team"]
    working.loc[missing_pitcher & away_batting, "opposing_pitcher_name"] = working.loc[
        missing_pitcher & away_batting, "home_probable_pitcher"
    ]
    working.loc[missing_pitcher & home_batting, "opposing_pitcher_name"] = working.loc[
        missing_pitcher & home_batting, "away_probable_pitcher"
    ]

    missing_throws = working["p_throws"].isna()
    working.loc[missing_throws & away_batting, "p_throws"] = working.loc[
        missing_throws & away_batting, "home_pitcher_throws"
    ]
    working.loc[missing_throws & home_batting, "p_throws"] = working.loc[
        missing_throws & home_batting, "away_pitcher_throws"
    ]
    return working


def prepare_book_level_odds_rows(
    raw_odds_df: pd.DataFrame,
    boost_book: str | None = None,
    boost_pct: float = 0.0,
) -> pd.DataFrame:
    if raw_odds_df.empty:
        return pd.DataFrame()

    odds_rows = raw_odds_df.copy()
    odds_rows = odds_rows[(odds_rows["side"] == "Over") & (odds_rows["point"] == 0.5)].copy()
    odds_rows["game_time"] = pd.to_datetime(odds_rows["game_time"], utc=True)
    odds_rows["game_date"] = (
        odds_rows["game_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    )
    odds_rows["game_time_utc"] = odds_rows["game_time"]
    odds_rows["home_team"] = odds_rows.apply(
        lambda row: normalize_team_name_to_abbr(row["home_team"], row["game_date"]),
        axis=1,
    )
    odds_rows["away_team"] = odds_rows.apply(
        lambda row: normalize_team_name_to_abbr(row["away_team"], row["game_date"]),
        axis=1,
    )
    odds_rows = odds_rows.dropna(subset=["home_team", "away_team"]).copy()
    odds_rows = add_scheduled_game_number(odds_rows)
    odds_rows["player_name_norm"] = (
        odds_rows["player"].apply(normalize_player_name).apply(apply_name_alias)
    )
    if "player_team_abbr" not in odds_rows.columns:
        odds_rows["player_team_abbr"] = np.nan

    odds_rows["effective_price"] = odds_rows["price"]
    if boost_book and boost_pct:
        boost_mask = odds_rows["bookmaker"].str.lower() == boost_book.lower()
        odds_rows.loc[boost_mask, "effective_price"] = odds_rows.loc[boost_mask, "price"].apply(
            lambda x: apply_profit_boost_to_odds(x, boost_pct)
        )

    odds_rows["effective_implied_prob"] = odds_rows["effective_price"].apply(american_to_prob)
    odds_rows["boost_applied"] = False
    if boost_book and boost_pct:
        odds_rows["boost_applied"] = odds_rows["bookmaker"].str.lower().eq(boost_book.lower())

    return odds_rows


def build_ranked_bets_table(
    scored_df: pd.DataFrame,
    raw_odds_df: pd.DataFrame,
    boost_book: str | None = None,
    boost_pct: float = 0.0,
) -> pd.DataFrame:
    if scored_df.empty or raw_odds_df.empty:
        return pd.DataFrame()

    odds_rows = prepare_book_level_odds_rows(
        raw_odds_df=raw_odds_df,
        boost_book=boost_book,
        boost_pct=boost_pct,
    )

    best_books = (
        odds_rows.sort_values(
            ["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm", "effective_price"],
            ascending=[True, True, True, True, True, False],
        )
        .groupby(["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm"], as_index=False)
        .head(1)
        .rename(
            columns={
                "player": "best_player_name",
                "player_team_abbr": "best_player_team_abbr",
                "bookmaker": "best_book",
                "bookmaker_title": "best_book_title",
                "price": "best_book_raw_odds",
                "effective_price": "best_book_odds",
                "effective_implied_prob": "best_book_implied_prob",
                "liquidity": "best_liquidity",
                "game_time_utc": "best_game_time_utc",
            }
        )
    )

    # Ensure best_liquidity exists even if the source data had no liquidity column
    if "best_liquidity" not in best_books.columns:
        best_books["best_liquidity"] = np.nan

    _best_books_cols = [
        "game_date",
        "home_team",
        "away_team",
        "scheduled_game_no",
        "player_name_norm",
        "best_player_name",
        "best_player_team_abbr",
        "best_game_time_utc",
        "best_book",
        "best_book_title",
        "best_book_raw_odds",
        "best_book_odds",
        "best_book_implied_prob",
        "best_liquidity",
    ]
    _best_books_cols = [c for c in _best_books_cols if c in best_books.columns]

    ranked = scored_df.merge(
        best_books[_best_books_cols],
        on=["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm"],
        how="left",
    )

    if "game_time_utc" not in ranked.columns and "best_game_time_utc" in ranked.columns:
        ranked["game_time_utc"] = ranked["best_game_time_utc"]
    if "game_time_utc" in ranked.columns:
        ranked["game_time_et"] = pd.to_datetime(ranked["game_time_utc"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    if "player_name" not in ranked.columns:
        ranked["player_name"] = np.nan
    if "best_player_name" in ranked.columns:
        ranked["player_name"] = ranked["player_name"].fillna(ranked["best_player_name"])
    if "batting_team" not in ranked.columns:
        ranked["batting_team"] = np.nan
    if "best_player_team_abbr" in ranked.columns:
        ranked["batting_team"] = ranked["batting_team"].fillna(ranked["best_player_team_abbr"])
    ranked = fill_pitcher_context_from_batting_team(ranked)

    ranked["edge_best_book"] = ranked["model_prob"] - ranked["best_book_implied_prob"]
    ranked["expected_profit_per_unit"] = (
        ranked["model_prob"] * ranked["best_book_odds"].apply(american_to_profit) -
        (1 - ranked["model_prob"])
    )
    ranked["boost_applied"] = (
        boost_book is not None and boost_pct > 0 and ranked["best_book"].str.lower().eq(boost_book.lower())
    )

    keep_cols = [
        "game_date",
        "game_time_et",
        "game_time_utc",
        "player_name",
        "home_team",
        "away_team",
        "batting_team",
        "opposing_pitcher_name",
        "p_throws",
        "temperature_f",
        "wind_speed_mph",
        "wind_direction",
        "best_book",
        "best_book_title",
        "best_book_raw_odds",
        "best_book_odds",
        "best_book_implied_prob",
        "best_liquidity",
        "boost_applied",
        "books_available",
        "model_prob",
        "market_implied_prob",
        "mean_implied_prob",
        "edge_best",
        "edge_best_book",
        "expected_profit_per_unit",
    ]

    for col in keep_cols:
        if col not in ranked.columns:
            ranked[col] = pd.NaT if col.endswith("_utc") or col.endswith("_et") else np.nan

    ranked = ranked[keep_cols].sort_values(
        ["edge_best_book", "expected_profit_per_unit", "model_prob"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return ranked


def build_all_loaded_bets_table(
    scored_df: pd.DataFrame,
    raw_odds_df: pd.DataFrame,
    boost_book: str | None = None,
    boost_pct: float = 0.0,
) -> pd.DataFrame:
    if scored_df.empty or raw_odds_df.empty:
        return pd.DataFrame()

    odds_rows = prepare_book_level_odds_rows(
        raw_odds_df=raw_odds_df,
        boost_book=boost_book,
        boost_pct=boost_pct,
    )

    if "liquidity" not in odds_rows.columns:
        odds_rows["liquidity"] = np.nan

    _all_bets_cols = [
        "game_date",
        "game_time_utc",
        "home_team",
        "away_team",
        "scheduled_game_no",
        "player_name_norm",
        "player",
        "player_team_abbr",
        "bookmaker",
        "bookmaker_title",
        "price",
        "effective_price",
        "effective_implied_prob",
        "liquidity",
        "boost_applied",
    ]
    _all_bets_cols = [c for c in _all_bets_cols if c in odds_rows.columns]

    all_bets = scored_df.merge(
        odds_rows[_all_bets_cols],
        on=["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm"],
        how="inner",
    )

    if "game_time_utc" in all_bets.columns:
        all_bets["game_time_et"] = pd.to_datetime(all_bets["game_time_utc"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    if "player_name" not in all_bets.columns:
        all_bets["player_name"] = np.nan
    if "player" in all_bets.columns:
        all_bets["player_name"] = all_bets["player_name"].fillna(all_bets["player"])
    if "batting_team" not in all_bets.columns:
        all_bets["batting_team"] = np.nan
    if "player_team_abbr" in all_bets.columns:
        all_bets["batting_team"] = all_bets["batting_team"].fillna(all_bets["player_team_abbr"])
    all_bets = fill_pitcher_context_from_batting_team(all_bets)

    all_bets["edge_vs_book"] = all_bets["model_prob"] - all_bets["effective_implied_prob"]
    all_bets["expected_profit_per_unit"] = (
        all_bets["model_prob"] * all_bets["effective_price"].apply(american_to_profit) -
        (1 - all_bets["model_prob"])
    )

    keep_cols = [
        "game_date",
        "game_time_et",
        "game_time_utc",
        "player_name",
        "home_team",
        "away_team",
        "batting_team",
        "opposing_pitcher_name",
        "bookmaker",
        "bookmaker_title",
        "price",
        "effective_price",
        "effective_implied_prob",
        "liquidity",
        "boost_applied",
        "books_available",
        "model_prob",
        "market_implied_prob",
        "mean_implied_prob",
        "edge_vs_book",
        "expected_profit_per_unit",
    ]

    for col in keep_cols:
        if col not in all_bets.columns:
            all_bets[col] = pd.NaT if col.endswith("_utc") or col.endswith("_et") else np.nan

    all_bets = all_bets[keep_cols].sort_values(
        ["edge_vs_book", "expected_profit_per_unit", "model_prob"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return all_bets


def format_bet_for_social(row: pd.Series, unit_size: float = 1.0, boost_pct: float = 0.0) -> str:
    odds_str = (
        f"{int(row['best_book_odds']):+d}"
        if pd.notna(row.get("best_book_odds"))
        else "N/A"
    )
    book = row.get("best_book_title") or row.get("Best Book") or row.get("best_book") or "book"
    player = row.get("player_name") or row.get("Player") or "Player"
    unit_str = f"{unit_size:.2f}".rstrip("0").rstrip(".")
    boosted = bool(row.get("boost_applied", False))
    if boosted and boost_pct > 0:
        return f"{player} to homer boosted {boost_pct:.0f}% to {odds_str} on {book} {unit_str} u"
    return f"{player} to homer {odds_str} on {book} {unit_str} u"

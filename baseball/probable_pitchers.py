import argparse
from pathlib import Path

import pandas as pd
import requests


SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_SPORT_ID = 1

TEAM_ABBR_NORMALIZATION = {
    "AZ": "ARI",
}


def normalize_team_abbr(team_abbr: str | None, game_date: pd.Timestamp | None = None) -> str | None:
    if team_abbr is None or pd.isna(team_abbr):
        return None

    team_abbr = str(team_abbr).strip().upper()
    year = pd.Timestamp(game_date).year if game_date is not None and not pd.isna(game_date) else None

    if team_abbr in TEAM_ABBR_NORMALIZATION:
        return TEAM_ABBR_NORMALIZATION[team_abbr]
    if team_abbr == "ATH":
        if year is not None and year <= 2024:
            return "OAK"
        return "ATH"

    return team_abbr


def _safe_request(params: dict, timeout: int = 30) -> dict:
    response = requests.get(SCHEDULE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _parse_side(game: dict, side: str, official_date: pd.Timestamp) -> dict:
    team_info = game.get("teams", {}).get(side, {})
    team_meta = team_info.get("team", {}) or {}
    probable_pitcher = team_info.get("probablePitcher", {}) or {}
    pitch_hand = probable_pitcher.get("pitchHand", {}) or {}
    opponent_side = "home" if side == "away" else "away"
    opponent_meta = game.get("teams", {}).get(opponent_side, {}).get("team", {}) or {}

    return {
        "game_date": official_date,
        "game_pk": game.get("gamePk"),
        "game_type": game.get("gameType"),
        "status": game.get("status", {}).get("detailedState"),
        "game_time_utc": pd.to_datetime(game.get("gameDate"), utc=True, errors="coerce"),
        "official_date": game.get("officialDate"),
        "doubleheader_code": game.get("doubleHeader"),
        "game_number": game.get("gameNumber"),
        "series_game_number": game.get("seriesGameNumber"),
        "day_night": game.get("dayNight"),
        "venue_name": (game.get("venue") or {}).get("name"),
        "team_side": side,
        "team_id": team_meta.get("id"),
        "team_name": team_meta.get("name"),
        "team_abbr": normalize_team_abbr(team_meta.get("abbreviation"), official_date),
        "opponent_id": opponent_meta.get("id"),
        "opponent_name": opponent_meta.get("name"),
        "opponent_abbr": normalize_team_abbr(opponent_meta.get("abbreviation"), official_date),
        "probable_pitcher_id": probable_pitcher.get("id"),
        "probable_pitcher_name": probable_pitcher.get("fullName"),
        "probable_pitcher_first_name": probable_pitcher.get("firstName"),
        "probable_pitcher_last_name": probable_pitcher.get("lastName"),
        "probable_pitcher_throws": pitch_hand.get("code"),
        "probable_pitcher_throws_desc": pitch_hand.get("description"),
        "is_probable_pitcher_confirmed": bool(probable_pitcher),
    }


def fetch_probable_pitchers(
    start_date: str,
    end_date: str | None = None,
    game_type: str | None = None,
    hydrate: str = "team,probablePitcher,person",
) -> pd.DataFrame:
    params = {
        "sportId": MLB_SPORT_ID,
        "hydrate": hydrate,
    }

    if end_date:
        params["startDate"] = start_date
        params["endDate"] = end_date
    else:
        params["date"] = start_date

    if game_type:
        params["gameType"] = game_type

    payload = _safe_request(params=params)

    rows = []
    for date_block in payload.get("dates", []):
        official_date = pd.to_datetime(date_block.get("date"), errors="coerce")
        for game in date_block.get("games", []):
            rows.append(_parse_side(game, "away", official_date))
            rows.append(_parse_side(game, "home", official_date))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        ["game_date", "game_time_utc", "game_pk", "team_side"]
    ).reset_index(drop=True)
    return df


def build_matchup_view(probables_df: pd.DataFrame) -> pd.DataFrame:
    if probables_df.empty:
        return probables_df.copy()

    away_df = (
        probables_df[probables_df["team_side"] == "away"]
        .rename(
            columns={
                "team_id": "away_team_id",
                "team_name": "away_team_name",
                "team_abbr": "away_team",
                "probable_pitcher_id": "away_pitcher_id",
                "probable_pitcher_name": "away_probable_pitcher",
                "probable_pitcher_throws": "away_pitcher_throws",
                "is_probable_pitcher_confirmed": "away_pitcher_confirmed",
            }
        )
        .copy()
    )

    home_df = (
        probables_df[probables_df["team_side"] == "home"]
        .rename(
            columns={
                "team_id": "home_team_id",
                "team_name": "home_team_name",
                "team_abbr": "home_team",
                "probable_pitcher_id": "home_pitcher_id",
                "probable_pitcher_name": "home_probable_pitcher",
                "probable_pitcher_throws": "home_pitcher_throws",
                "is_probable_pitcher_confirmed": "home_pitcher_confirmed",
            }
        )
        .copy()
    )

    matchup_df = away_df.merge(
        home_df,
        on=[
            "game_date",
            "game_pk",
            "game_type",
            "status",
            "game_time_utc",
            "official_date",
            "doubleheader_code",
            "game_number",
            "series_game_number",
            "day_night",
            "venue_name",
        ],
        how="outer",
    )

    keep_cols = [
        "game_date",
        "game_time_utc",
        "game_pk",
        "game_type",
        "status",
        "doubleheader_code",
        "game_number",
        "series_game_number",
        "day_night",
        "venue_name",
        "away_team_id",
        "away_team",
        "away_team_name",
        "away_probable_pitcher",
        "away_pitcher_id",
        "away_pitcher_throws",
        "away_pitcher_confirmed",
        "home_team_id",
        "home_team",
        "home_team_name",
        "home_probable_pitcher",
        "home_pitcher_id",
        "home_pitcher_throws",
        "home_pitcher_confirmed",
    ]
    return matchup_df[keep_cols].sort_values(
        ["game_date", "game_time_utc", "game_pk"]
    ).reset_index(drop=True)


def _write_output(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return
    raise ValueError("Output path must end with .csv or .parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch MLB probable pitchers from the MLB Stats API."
    )
    parser.add_argument(
        "--date",
        default=pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"),
        help="Single date in YYYY-MM-DD format. Defaults to today's ET date.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date in YYYY-MM-DD format for a date range.",
    )
    parser.add_argument(
        "--game-type",
        help="Optional MLB game type filter, for example R, S, or P.",
    )
    parser.add_argument(
        "--matchups",
        action="store_true",
        help="Return one row per game with away/home probable pitchers side by side.",
    )
    parser.add_argument(
        "--output",
        help="Optional .csv or .parquet output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probables_df = fetch_probable_pitchers(
        start_date=args.date,
        end_date=args.end_date,
        game_type=args.game_type,
    )

    if args.matchups:
        probables_df = build_matchup_view(probables_df)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_output(probables_df, output_path)
        print(f"Saved {len(probables_df):,} rows to {output_path}")
        return

    if probables_df.empty:
        print("No games found for the requested date range.")
        return

    print(probables_df.to_string(index=False))


if __name__ == "__main__":
    main()

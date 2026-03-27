import argparse
from pathlib import Path

import pandas as pd
import requests

from build_weather import DOME_TEAMS, PARK_COORDS, wind_direction_to_label
from probable_pitchers import build_matchup_view, fetch_probable_pitchers


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def add_scheduled_game_number(matchups_df: pd.DataFrame) -> pd.DataFrame:
    matchups_df = matchups_df.copy()
    matchups_df["game_date"] = pd.to_datetime(matchups_df["game_date"]).dt.normalize()
    matchups_df = matchups_df.sort_values(
        ["game_date", "home_team", "away_team", "game_time_utc", "game_pk"]
    ).reset_index(drop=True)
    matchups_df["scheduled_game_no"] = (
        matchups_df.groupby(["game_date", "home_team", "away_team"]).cumcount() + 1
    )
    return matchups_df


def fetch_open_meteo_forecast_range(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timeout: int = 30,
) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "America/New_York",
    }

    response = requests.get(FORECAST_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})

    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    forecast_df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temperature_f": pd.Series(hourly["temperature_2m"], dtype="float64") * 9 / 5 + 32,
        "humidity": pd.Series(hourly["relative_humidity_2m"], dtype="float64"),
        "wind_speed_mph": pd.Series(hourly["wind_speed_10m"], dtype="float64") * 0.621371,
        "wind_direction_deg": pd.Series(hourly["wind_direction_10m"], dtype="float64"),
    })
    forecast_df["game_date"] = forecast_df["time"].dt.normalize()
    forecast_df["game_hour"] = forecast_df["time"].dt.hour
    return forecast_df


def build_live_games(
    start_date: str,
    end_date: str | None = None,
    game_type: str | None = None,
) -> pd.DataFrame:
    probables_df = fetch_probable_pitchers(
        start_date=start_date,
        end_date=end_date,
        game_type=game_type,
    )
    matchups_df = build_matchup_view(probables_df)

    if matchups_df.empty:
        return matchups_df

    matchups_df = add_scheduled_game_number(matchups_df)
    matchups_df["game_time_utc"] = pd.to_datetime(matchups_df["game_time_utc"], utc=True)
    matchups_df["game_time_et"] = matchups_df["game_time_utc"].dt.tz_convert("America/New_York")
    matchups_df["game_date"] = matchups_df["game_time_et"].dt.tz_localize(None).dt.normalize()
    matchups_df["game_hour"] = matchups_df["game_time_et"].dt.hour
    return matchups_df


def build_live_weather_df(
    start_date: str,
    end_date: str | None = None,
    game_type: str | None = None,
) -> pd.DataFrame:
    games = build_live_games(
        start_date=start_date,
        end_date=end_date,
        game_type=game_type,
    )

    if games.empty:
        return pd.DataFrame()

    weather_parts = []
    for home_team, team_games in games.groupby("home_team"):
        if home_team not in PARK_COORDS:
            continue

        coords = PARK_COORDS[home_team]
        start = team_games["game_date"].min().strftime("%Y-%m-%d")
        end = team_games["game_date"].max().strftime("%Y-%m-%d")
        forecast_df = fetch_open_meteo_forecast_range(
            lat=coords["lat"],
            lon=coords["lon"],
            start_date=start,
            end_date=end,
        )

        if forecast_df.empty:
            continue

        lookup = team_games[[
            "game_pk",
            "game_date",
            "game_time_utc",
            "game_time_et",
            "home_team",
            "away_team",
            "scheduled_game_no",
            "venue_name",
            "status",
            "game_hour",
        ]].copy()

        merged = lookup.merge(
            forecast_df,
            on=["game_date", "game_hour"],
            how="left",
        )
        merged["wind_direction"] = merged["wind_direction_deg"].apply(wind_direction_to_label)
        merged["roof_closed"] = int(home_team in DOME_TEAMS)
        weather_parts.append(merged)

    if not weather_parts:
        return pd.DataFrame()

    weather_df = pd.concat(weather_parts, ignore_index=True).drop_duplicates()
    weather_df = weather_df.sort_values(
        ["game_date", "game_time_utc", "game_pk"]
    ).reset_index(drop=True)
    return weather_df[[
        "game_pk",
        "game_date",
        "game_time_utc",
        "game_time_et",
        "status",
        "venue_name",
        "home_team",
        "away_team",
        "scheduled_game_no",
        "temperature_f",
        "humidity",
        "wind_speed_mph",
        "wind_direction_deg",
        "wind_direction",
        "roof_closed",
    ]]


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
        description="Fetch live MLB weather forecasts in model-ready format."
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
        "--output",
        help="Optional .csv or .parquet output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weather_df = build_live_weather_df(
        start_date=args.date,
        end_date=args.end_date,
        game_type=args.game_type,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_output(weather_df, output_path)
        print(f"Saved {len(weather_df):,} rows to {output_path}")
        return

    if weather_df.empty:
        print("No live weather rows found for the requested date range.")
        return

    print(weather_df.to_string(index=False))


if __name__ == "__main__":
    main()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from load import load_hr_odds_parquets


TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SF",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

PARK_COORDS = {
    "ARI": {"lat": 33.4453, "lon": -112.0667},
    "ATH": {"lat": 38.5805, "lon": -121.5139},
    "ATL": {"lat": 33.8908, "lon": -84.4677},
    "BAL": {"lat": 39.2838, "lon": -76.6217},
    "BOS": {"lat": 42.3467, "lon": -71.0972},
    "CHC": {"lat": 41.9484, "lon": -87.6553},
    "CWS": {"lat": 41.8300, "lon": -87.6339},
    "CIN": {"lat": 39.0979, "lon": -84.5081},
    "CLE": {"lat": 41.4962, "lon": -81.6852},
    "COL": {"lat": 39.7559, "lon": -104.9942},
    "DET": {"lat": 42.3390, "lon": -83.0485},
    "HOU": {"lat": 29.7573, "lon": -95.3555},
    "KC": {"lat": 39.0517, "lon": -94.4803},
    "LAA": {"lat": 33.8003, "lon": -117.8827},
    "LAD": {"lat": 34.0739, "lon": -118.2400},
    "MIA": {"lat": 25.7781, "lon": -80.2197},
    "MIL": {"lat": 43.0280, "lon": -87.9712},
    "MIN": {"lat": 44.9817, "lon": -93.2778},
    "NYM": {"lat": 40.7571, "lon": -73.8458},
    "NYY": {"lat": 40.8296, "lon": -73.9262},
    "OAK": {"lat": 37.7516, "lon": -122.2005},
    "PHI": {"lat": 39.9061, "lon": -75.1665},
    "PIT": {"lat": 40.4469, "lon": -80.0057},
    "SD": {"lat": 32.7073, "lon": -117.1566},
    "SEA": {"lat": 47.5914, "lon": -122.3325},
    "SF": {"lat": 37.7786, "lon": -122.3893},
    "STL": {"lat": 38.6226, "lon": -90.1928},
    "TB": {"lat": 27.7682, "lon": -82.6534},
    "TEX": {"lat": 32.7473, "lon": -97.0847},
    "TOR": {"lat": 43.6414, "lon": -79.3894},
    "WSH": {"lat": 38.8730, "lon": -77.0074},
}

DOME_TEAMS = {"ARI", "HOU", "MIA", "MIL", "SEA", "TB", "TEX", "TOR"}


def normalize_team_name_to_abbr(team_name: str, game_date: pd.Timestamp | None = None) -> str | None:
    if pd.isna(team_name):
        return None

    team_name = str(team_name).strip()
    year = pd.Timestamp(game_date).year if game_date is not None and not pd.isna(game_date) else None

    if team_name == "Arizona Diamondbacks":
        return "ARI"
    if team_name in {"Athletics", "Oakland Athletics"}:
        if year is not None and year <= 2024:
            return "OAK"
        return "ATH"

    mapped = TEAM_NAME_TO_ABBR.get(team_name)
    if mapped == "AZ":
        return "ARI"
    return mapped


def load_games_from_odds_parquets(
    folder_path: str,
    side: str = "Over",
    point: float = 0.5,
) -> pd.DataFrame:
    odds_df = load_hr_odds_parquets(
        folder_path=folder_path,
        side=side,
        point=point,
        verbose=True,
    )

    games = (
        odds_df[["game_id", "game_time", "home_team", "away_team"]]
        .drop_duplicates()
        .copy()
    )

    games["home_team_name"] = games["home_team"]
    games["away_team_name"] = games["away_team"]

    games["game_time"] = pd.to_datetime(games["game_time"], utc=True)
    games["game_time_et"] = games["game_time"].dt.tz_convert("America/New_York")
    games["game_date"] = games["game_time_et"].dt.normalize()

    games["home_team"] = games.apply(
        lambda row: normalize_team_name_to_abbr(
            row["home_team_name"],
            row["game_date"],
        ),
        axis=1,
    )
    games["away_team"] = games.apply(
        lambda row: normalize_team_name_to_abbr(
            row["away_team_name"],
            row["game_date"],
        ),
        axis=1,
    )

    unmapped_home = games.loc[games["home_team"].isna(), "home_team_name"].dropna().unique()
    unmapped_away = games.loc[games["away_team"].isna(), "away_team_name"].dropna().unique()

    if len(unmapped_home) > 0:
        print("Unmapped home team names:", sorted(unmapped_home))
    if len(unmapped_away) > 0:
        print("Unmapped away team names:", sorted(unmapped_away))

    games = games.dropna(subset=["home_team", "away_team"]).copy()

    games["game_hour"] = games["game_time_et"].dt.hour
    games = games.sort_values(["game_date", "home_team", "away_team", "game_time", "game_id"]).reset_index(drop=True)
    games["scheduled_game_no"] = (
        games.groupby(["game_date", "home_team", "away_team"])
        .cumcount() + 1
    )

    return games


def fetch_open_meteo_hourly_range(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    pause_seconds: float = 0.25,
    max_retries: int = 6,
) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
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

    for attempt in range(max_retries):
        try:
            time.sleep(pause_seconds)
            r = requests.get(url, params=params, timeout=60)

            if r.status_code == 429:
                wait = min(2 ** attempt, 60)
                print(f"429 received for {start_date} to {end_date}. Sleeping {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            payload = r.json()
            hourly = payload.get("hourly", {})

            if not hourly or "time" not in hourly:
                return pd.DataFrame()

            df = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"]),
                "temperature_f": pd.Series(hourly["temperature_2m"], dtype="float64") * 9 / 5 + 32,
                "humidity": pd.Series(hourly["relative_humidity_2m"], dtype="float64"),
                "wind_speed_mph": pd.Series(hourly["wind_speed_10m"], dtype="float64") * 0.621371,
                "wind_direction_deg": pd.Series(hourly["wind_direction_10m"], dtype="float64"),
            })

            df["game_date"] = df["time"].dt.normalize()
            df["game_hour"] = df["time"].dt.hour
            return df

        except requests.exceptions.RequestException as e:
            wait = min(2 ** attempt, 60)
            print(f"Request failed: {e}. Sleeping {wait}s...")
            time.sleep(wait)

    print(f"Failed after retries for {start_date} to {end_date}")
    return pd.DataFrame()


def wind_direction_to_label(deg: float) -> str | None:
    if pd.isna(deg):
        return None
    if 45 <= deg < 135:
        return "cross"
    if 135 <= deg < 225:
        return "in"
    if 225 <= deg < 315:
        return "cross"
    return "out"


def build_team_weather_lookup(
    team: str,
    team_games: pd.DataFrame,
    max_days_per_request: int = 30,
) -> pd.DataFrame:
    if team not in PARK_COORDS:
        return pd.DataFrame()

    coords = PARK_COORDS[team]

    unique_dates = (
        pd.Series(team_games["game_date"].drop_duplicates().sort_values().tolist())
        .dt.tz_localize(None)
        .sort_values()
        .reset_index(drop=True)
    )

    if unique_dates.empty:
        return pd.DataFrame()

    weather_chunks = []

    for start_idx in range(0, len(unique_dates), max_days_per_request):
        chunk_dates = unique_dates.iloc[start_idx:start_idx + max_days_per_request]
        start_date = chunk_dates.min().strftime("%Y-%m-%d")
        end_date = chunk_dates.max().strftime("%Y-%m-%d")

        hourly_df = fetch_open_meteo_hourly_range(
            lat=coords["lat"],
            lon=coords["lon"],
            start_date=start_date,
            end_date=end_date,
        )

        if not hourly_df.empty:
            weather_chunks.append(hourly_df)

    if not weather_chunks:
        return pd.DataFrame()

    team_weather = pd.concat(weather_chunks, ignore_index=True).drop_duplicates()

    lookup = team_games[[
        "game_id",
        "game_date",
        "game_hour",
        "home_team",
        "away_team",
        "scheduled_game_no",
    ]].copy()
    lookup["game_date_naive"] = lookup["game_date"].dt.tz_localize(None)

    merged = lookup.merge(
        team_weather,
        left_on=["game_date_naive", "game_hour"],
        right_on=["game_date", "game_hour"],
        how="left",
    )

    merged["wind_direction"] = merged["wind_direction_deg"].apply(wind_direction_to_label)
    merged["roof_closed"] = (merged["home_team"].isin(DOME_TEAMS)).astype(int)

    result = merged[[
        "game_id",
        "game_date_x",
        "home_team",
        "away_team",
        "scheduled_game_no",
        "temperature_f",
        "humidity",
        "wind_speed_mph",
        "wind_direction_deg",
        "wind_direction",
        "roof_closed",
    ]].rename(columns={"game_date_x": "game_date"})

    return result.drop_duplicates()


def save_checkpoint(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved checkpoint: {path} | rows={len(df)}")


def build_weather_df_from_odds_parquets(
    folder_path: str,
    side: str = "Over",
    point: float = 0.5,
    checkpoint_dir: str = "weather_checkpoints",
) -> pd.DataFrame:
    games = load_games_from_odds_parquets(
        folder_path=folder_path,
        side=side,
        point=point,
    )

    print("Games after team mapping:", games.shape)
    if not games.empty:
        print(games[["home_team_name", "home_team"]].drop_duplicates().head(10))

    games = games.sort_values(["home_team", "game_date", "game_hour"]).reset_index(drop=True)

    all_weather_parts = []
    teams = [t for t in games["home_team"].dropna().unique() if t in PARK_COORDS]

    for i, team in enumerate(tqdm(teams, desc="Fetching weather by team"), start=1):
        team_games = games.loc[games["home_team"] == team].copy()

        team_result = build_team_weather_lookup(
            team=team,
            team_games=team_games,
            max_days_per_request=30,
        )

        if not team_result.empty:
            all_weather_parts.append(team_result)

        partial_df = (
            pd.concat(all_weather_parts, ignore_index=True).drop_duplicates()
            if all_weather_parts
            else pd.DataFrame(columns=[
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "scheduled_game_no",
                "temperature_f",
                "humidity",
                "wind_speed_mph",
                "wind_direction_deg",
                "wind_direction",
                "roof_closed",
            ])
        )

        save_checkpoint(
            partial_df,
            f"{checkpoint_dir}/weather_checkpoint_team_{i:02d}_{team}.parquet"
        )

        save_checkpoint(
            partial_df,
            f"{checkpoint_dir}/weather_checkpoint_latest.parquet"
        )

    weather_df = (
        pd.concat(all_weather_parts, ignore_index=True).drop_duplicates()
        if all_weather_parts
        else pd.DataFrame()
    )

    return weather_df


if __name__ == "__main__":
    weather_df = build_weather_df_from_odds_parquets(
        folder_path="mlb_hr_odds_chunks",
        side="Over",
        point=0.5,
        checkpoint_dir="weather_checkpoints",
    )

    weather_df.to_parquet("weather_from_odds.parquet", index=False)
    print(weather_df.head())
    print(weather_df.shape)

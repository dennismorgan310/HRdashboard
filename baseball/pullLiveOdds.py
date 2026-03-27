import argparse
import os
import time
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import requests


API_KEY_PATH = Path("apiKey.txt")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
DEFAULT_REGIONS = "us"
DEFAULT_MARKET = "batter_home_runs"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_OUTPUT_DIR = Path("live_odds")


def _read_api_key_from_streamlit_secrets() -> str | None:
    try:
        import streamlit as st
    except Exception:
        return None

    for key in ("ODDS_API_KEY", "THE_ODDS_API_KEY", "odds_api_key", "the_odds_api_key"):
        try:
            value = st.secrets[key]
        except Exception:
            value = None
        if value:
            return str(value).strip()
    return None


def read_api_key(path: Path = API_KEY_PATH) -> str:
    for env_key in ("ODDS_API_KEY", "THE_ODDS_API_KEY"):
        value = os.getenv(env_key)
        if value and value.strip():
            return value.strip()

    secret_value = _read_api_key_from_streamlit_secrets()
    if secret_value:
        return secret_value

    if not path.exists():
        raise FileNotFoundError(
            f"API key file not found: {path}. Set ODDS_API_KEY/THE_ODDS_API_KEY or provide the key in st.secrets."
        )
    return path.read_text().strip()


def safe_get(url: str, params: dict, timeout: int = 30, max_retries: int = 3) -> dict | list:
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code in {402, 429}:
                raise RuntimeError(
                    f"Odds API quota/rate limit reached (status {response.status_code}). "
                    f"x-requests-last={response.headers.get('x-requests-last')}, "
                    f"x-requests-used={response.headers.get('x-requests-used')}, "
                    f"x-requests-remaining={response.headers.get('x-requests-remaining')}"
                )

            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == max_retries:
                raise
            sleep_s = 2 ** attempt
            time.sleep(sleep_s)


def get_live_events(
    api_key: str,
    commence_time_from: str | None = None,
    event_ids: str | None = None,
) -> list[dict]:
    url = f"{BASE_URL}/sports/{SPORT}/events"
    params = {
        "apiKey": api_key,
        "dateFormat": "iso",
    }
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if event_ids:
        params["eventIds"] = event_ids
    payload = safe_get(url, params=params)
    return payload if isinstance(payload, list) else []


def get_live_event_odds(
    api_key: str,
    event_id: str,
    regions: str,
    markets: str,
    odds_format: str,
    bookmakers: str | None = None,
) -> dict:
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    payload = safe_get(url, params=params)
    return payload if isinstance(payload, dict) else {}


def parse_event_markets(
    event_data: dict,
    snapshot_time: str,
    snapshot_label: str,
    requested_markets: set[str],
) -> list[dict]:
    rows = []

    for bookmaker in event_data.get("bookmakers", []):
        bookmaker_key = bookmaker.get("key")
        bookmaker_title = bookmaker.get("title")
        bookmaker_last_update = bookmaker.get("last_update")

        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            if market_key not in requested_markets:
                continue

            market_last_update = market.get("last_update")

            for outcome in market.get("outcomes", []):
                rows.append(
                    {
                        "snapshot_time": snapshot_time,
                        "snapshot_label": snapshot_label,
                        "game_id": event_data.get("id"),
                        "game_time": event_data.get("commence_time"),
                        "home_team": event_data.get("home_team"),
                        "away_team": event_data.get("away_team"),
                        "bookmaker": bookmaker_key,
                        "bookmaker_title": bookmaker_title,
                        "market_key": market_key,
                        "player": outcome.get("description"),
                        "side": outcome.get("name"),
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                        "market_last_update": market_last_update,
                        "book_last_update": bookmaker_last_update,
                    }
                )

    return rows


def american_to_prob(odds):
    if pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def clean_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[
        (df["side"] == "Over") &
        (pd.to_numeric(df["point"], errors="coerce") == 0.5)
    ].copy()

    if df.empty:
        return df

    df["implied_prob"] = df["price"].apply(american_to_prob)
    df = df.drop_duplicates(
        subset=[
            "snapshot_time",
            "snapshot_label",
            "game_id",
            "bookmaker",
            "player",
            "side",
            "point",
            "price",
        ]
    ).reset_index(drop=True)
    return df


def fetch_live_hr_odds(
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKET,
    odds_format: str = DEFAULT_ODDS_FORMAT,
    bookmakers: str | None = None,
    snapshot_label: str = "live",
    target_date: date | None = None,
    commence_time_from: str | None = None,
    event_ids: str | None = None,
    api_key_path: Path = API_KEY_PATH,
) -> pd.DataFrame:
    api_key = read_api_key(api_key_path)
    snapshot_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    requested_markets = {
        market.strip() for market in markets.split(",") if market.strip()
    }

    events = get_live_events(
        api_key=api_key,
        commence_time_from=commence_time_from,
        event_ids=event_ids,
    )

    if target_date is not None:
        filtered_events = []
        for event in events:
            commence_time = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
            if pd.isna(commence_time):
                continue
            event_game_date = commence_time.tz_convert("America/New_York").date()
            if event_game_date == target_date:
                filtered_events.append(event)
        events = filtered_events

    all_rows = []
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        event_data = get_live_event_odds(
            api_key=api_key,
            event_id=event_id,
            regions=regions,
            markets=markets,
            odds_format=odds_format,
            bookmakers=bookmakers,
        )
        all_rows.extend(
            parse_event_markets(
                event_data=event_data,
                snapshot_time=snapshot_time,
                snapshot_label=snapshot_label,
                requested_markets=requested_markets,
            )
        )

    return clean_rows(all_rows)


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return
    raise ValueError("Output path must end with .csv or .parquet")


def default_output_path(output_dir: Path, snapshot_label: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"mlb_hr_live_odds_{snapshot_label}_{timestamp}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull live MLB home run prop odds from The Odds API."
    )
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="Comma-separated Odds API regions.")
    parser.add_argument("--markets", default=DEFAULT_MARKET, help="Comma-separated Odds API markets.")
    parser.add_argument("--odds-format", default=DEFAULT_ODDS_FORMAT, choices=["american", "decimal"])
    parser.add_argument("--bookmakers", help="Optional comma-separated bookmaker keys, for example novig,prophetx.")
    parser.add_argument("--snapshot-label", default="live", help="Snapshot label stored in the output.")
    parser.add_argument(
        "--date",
        help="Optional ET game date filter in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--commence-time-from",
        help="Optional ISO8601 UTC timestamp to filter events on or after a given time.",
    )
    parser.add_argument("--event-ids", help="Optional comma-separated event ids.")
    parser.add_argument("--api-key-path", default=str(API_KEY_PATH), help="Path to api key text file.")
    parser.add_argument("--output", help="Optional .csv or .parquet output path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Used only when --output is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = fetch_live_hr_odds(
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
        bookmakers=args.bookmakers,
        snapshot_label=args.snapshot_label,
        target_date=date.fromisoformat(args.date) if args.date else None,
        commence_time_from=args.commence_time_from,
        event_ids=args.event_ids,
        api_key_path=Path(args.api_key_path),
    )

    if df.empty:
        print("No live odds rows found.")
        return

    output_path = Path(args.output) if args.output else default_output_path(
        output_dir=Path(args.output_dir),
        snapshot_label=args.snapshot_label,
    )
    write_output(df, output_path)
    print(f"Saved {len(df):,} rows to {output_path}")
    print(
        df[[
            "game_id",
            "game_time",
            "away_team",
            "home_team",
            "bookmaker",
            "player",
            "side",
            "price",
            "point",
        ]].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()

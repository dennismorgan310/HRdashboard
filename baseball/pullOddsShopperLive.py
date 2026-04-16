import argparse
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from oddshopper_auth import (
    CHROME_BINARY_PATH,
    COOKIES_PATH,
    LOGIN_INFO_PATH,
    LOGIN_URL,
    build_authenticated_session,
    load_login_info,
)


BASE_URL = "https://www.oddsshopper.com"
DEFAULT_LEAGUE = "MLB"
DEFAULT_STATE = "PA"
DEFAULT_MARKET_NAME = "Total Home Runs"
DEFAULT_OUTPUT_DIR = Path("live_odds")
LIVE_ODDS_PAGE_URL = "https://www.oddsshopper.com/liveodds/mlb/playerprops/Total-Home-Runs"
COOKIE_HEADER_PATH = Path("oddshopper_cookie_header.txt")


def _get_streamlit_secret(key: str):
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


def get_oddsshopper_cookie_header(
    path: Path = COOKIE_HEADER_PATH,
    override: str | None = None,
) -> str | None:
    value = override or os.getenv("ODDSHOPPER_COOKIE_HEADER") or _get_streamlit_secret("ODDSHOPPER_COOKIE_HEADER")
    if not value:
        value = load_cookie_header(path)
    if not value:
        return None
    value = str(value).strip()
    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()
    return value or None


def get_oddsshopper_auth_diagnostics(path: Path = COOKIE_HEADER_PATH) -> dict:
    env_present = bool(os.getenv("ODDSHOPPER_COOKIE_HEADER", "").strip())
    secret_present = bool(str(_get_streamlit_secret("ODDSHOPPER_COOKIE_HEADER") or "").strip())
    file_present = path.exists() and bool(path.read_text().strip())
    resolved_source = (
        "env" if env_present else
        "streamlit_secrets" if secret_present else
        "file" if file_present else
        "missing"
    )
    return {
        "resolved_source": resolved_source,
        "env_present": env_present,
        "secret_present": secret_present,
        "file_present": file_present,
    }


def american_to_prob(odds):
    if pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def _safe_get_json(url: str, timeout: int = 30, session: requests.Session | None = None) -> dict:
    request_session = session or requests.Session()
    request_session.headers.update({"User-Agent": "Mozilla/5.0"})
    response = request_session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_cookie_header(path: Path = COOKIE_HEADER_PATH) -> str | None:
    if not path.exists():
        return None
    value = path.read_text().strip()
    if not value:
        return None
    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()
    return value or None


def _to_book_key(sportsbook_code: str | None) -> str | None:
    if sportsbook_code is None:
        return None
    return (
        str(sportsbook_code)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def get_offer_id(
    league: str = DEFAULT_LEAGUE,
    market_name: str = DEFAULT_MARKET_NAME,
    session: requests.Session | None = None,
) -> str:
    payload = _safe_get_json(f"{BASE_URL}/api/liveOdds/offers?league={league}", session=session)
    for category in payload.get("offerCategories", []):
        for offer in category.get("offers", []):
            if str(offer.get("name", "")).strip().lower() == market_name.strip().lower():
                return offer["id"]
    raise ValueError(f"Could not find offerId for {league=} {market_name=}")


def build_date_window_et(target_date: date) -> tuple[str, str]:
    start_et = datetime.combine(target_date, time.min).replace(tzinfo=timezone(timedelta(hours=-4)))
    end_et = datetime.combine(target_date, time.max).replace(tzinfo=timezone(timedelta(hours=-4)))
    start_utc = start_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_utc = end_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.999Z")
    return start_utc, end_utc


def _parse_american_odds(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("+", "")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_liquidity(book: dict) -> float | None:
    for key in [
        "availableLiquidity",
        "liquidity",
        "available_liquidity",
        "maxStake",
        "maxWager",
        "maxBet",
        "availableToBet",
        "limit",
        "volume",
    ]:
        value = book.get(key)
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _split_event_name(event_name: str) -> tuple[str | None, str | None]:
    if " @ " not in event_name:
        return None, None
    away, home = event_name.split(" @ ", 1)
    return home.strip(), away.strip()


def _build_chrome_driver() -> webdriver.Chrome:
    options = ChromeOptions()
    if CHROME_BINARY_PATH.exists():
        options.binary_location = str(CHROME_BINARY_PATH)
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1200")
    return webdriver.Chrome(options=options)


def _login_driver(
    driver: webdriver.Chrome,
    login_info_path: Path,
    login_url: str,
    wait_seconds: int = 45,
) -> None:
    email, password = load_login_info(login_info_path)
    driver.get(login_url)
    wait = WebDriverWait(driver, wait_seconds)

    email_input = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='username']"))
    )
    email_input.clear()
    email_input.send_keys(email)

    password_input = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password'], input[name='password']"))
    )
    password_input.clear()
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)

    wait.until(lambda d: "oddsshopper.com" in d.current_url and "/api/auth/callback" not in d.current_url)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))


def _fetch_json_via_browser_session(
    api_url: str,
    state: str,
    login_info_path: Path,
    login_url: str,
    wait_seconds: int = 45,
) -> dict:
    driver = _build_chrome_driver()
    try:
        _login_driver(driver, login_info_path=login_info_path, login_url=login_url, wait_seconds=wait_seconds)
        driver.get(LIVE_ODDS_PAGE_URL)
        WebDriverWait(driver, wait_seconds).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        driver.execute_script(
            """
            document.cookie = arguments[0] + "; path=/; domain=.oddsshopper.com";
            document.cookie = arguments[1] + "; path=/; domain=.oddsshopper.com";
            """,
            f"userLocation={json.dumps({'state': state, 'timeZone': 'America/New_York'})}",
            "ismobileapp=false",
        )
        payload = driver.execute_async_script(
            """
            const url = arguments[0];
            const done = arguments[arguments.length - 1];
            fetch(url, {
              method: 'GET',
              credentials: 'include',
              headers: {
                'accept': '*/*',
                'accept-language': 'en-US,en;q=0.9,es;q=0.8,it;q=0.7'
              }
            })
              .then(async (response) => {
                const text = await response.text();
                done({
                  ok: response.ok,
                  status: response.status,
                  text
                });
              })
              .catch((error) => done({ ok: false, status: 0, error: String(error) }));
            """,
            api_url,
        )
        if not payload.get("ok"):
            raise RuntimeError(
                f"Browser session fetch failed with status={payload.get('status')}: {payload.get('error') or payload.get('text')}"
            )
        return json.loads(payload["text"])
    except TimeoutException as exc:
        raise RuntimeError("Timed out during browser-session OddsShopper fetch.") from exc
    finally:
        driver.quit()


def fetch_oddsshopper_live_hr_odds(
    target_date: date,
    state: str = DEFAULT_STATE,
    league: str = DEFAULT_LEAGUE,
    market_name: str = DEFAULT_MARKET_NAME,
    snapshot_label: str = "live_oddsshopper",
    sportsbook_filter: str | None = None,
    cookies_path: Path = COOKIES_PATH,
    cookie_header_path: Path = COOKIE_HEADER_PATH,
    cookie_header_override: str | None = None,
    use_browser_session: bool = False,
    login_info_path: Path = LOGIN_INFO_PATH,
    login_url: str = LOGIN_URL,
) -> pd.DataFrame:
    session = None
    if not use_browser_session:
        session = build_authenticated_session(cookies_path) if cookies_path.exists() else requests.Session()
        session.headers.update({
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8,it;q=0.7",
            "Origin": BASE_URL,
            "Referer": LIVE_ODDS_PAGE_URL,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
            ),
        })
        raw_cookie_header = get_oddsshopper_cookie_header(
            path=cookie_header_path,
            override=cookie_header_override,
        )
        if raw_cookie_header:
            session.headers["Cookie"] = raw_cookie_header
        else:
            session.cookies.set(
                "userLocation",
                json.dumps({"state": state, "timeZone": "America/New_York"}),
                domain="www.oddsshopper.com",
                path="/",
            )
        session.get(LIVE_ODDS_PAGE_URL, timeout=30)

    offer_id = get_offer_id(league=league, market_name=market_name, session=session)
    start_utc, end_utc = build_date_window_et(target_date)
    api_url = (
        f"{BASE_URL}/api/liveOdds/odds?offerId={offer_id}"
        f"&state={state}&startDate={start_utc}&endDate={end_utc}"
    )
    if use_browser_session:
        payload = _fetch_json_via_browser_session(
            api_url=api_url,
            state=state,
            login_info_path=login_info_path,
            login_url=login_url,
        )
    else:
        payload = _safe_get_json(api_url, session=session)

    allowed_books = None
    if sportsbook_filter:
        allowed_books = {
            _to_book_key(book) for book in sportsbook_filter.split(",") if str(book).strip()
        }

    rows = []
    snapshot_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for event in payload.get("events", []):
        home_team, away_team = _split_event_name(event.get("eventName", ""))
        game_time = pd.to_datetime(event.get("startDate"), utc=True, errors="coerce")
        game_time_str = game_time.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(game_time) else None

        for participant in event.get("participants", []):
            player_name = participant.get("name")
            player_team_abbr = participant.get("abbreviation")

            for side in event.get("sides", []):
                side_label = side.get("label")
                if side_label != "Over":
                    continue

                for book in side.get("books", []):
                    book_key = _to_book_key(book.get("sportsbookCode"))
                    if allowed_books is not None and book_key not in allowed_books:
                        continue

                    price = _parse_american_odds(book.get("americanOdds"))
                    line = pd.to_numeric(book.get("line"), errors="coerce")

                    if pd.isna(price) or pd.isna(line) or float(line) != 0.5:
                        continue

                    rows.append(
                        {
                            "snapshot_time": snapshot_time,
                            "snapshot_label": snapshot_label,
                            "game_id": event.get("eventId"),
                            "game_time": game_time_str,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": book_key,
                            "bookmaker_title": book.get("sportsbookCode"),
                            "market_key": "batter_home_runs",
                            "player": player_name,
                            "player_team_abbr": player_team_abbr,
                            "side": side_label,
                            "price": price,
                            "point": line,
                            "liquidity": _parse_liquidity(book),
                            "market_last_update": None,
                            "book_last_update": None,
                            "deep_link_url": book.get("deepLinkUrl"),
                            "implied_prob": american_to_prob(price),
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=[
            "game_id",
            "bookmaker",
            "player",
            "side",
            "point",
            "price",
        ]
    ).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull live MLB home run prop odds from OddsShopper."
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="ET game date in YYYY-MM-DD.")
    parser.add_argument("--state", default=DEFAULT_STATE, help="OddsShopper state code, for example PA.")
    parser.add_argument("--league", default=DEFAULT_LEAGUE)
    parser.add_argument("--market-name", default=DEFAULT_MARKET_NAME)
    parser.add_argument("--bookmakers", help="Optional comma-separated sportsbook filter.")
    parser.add_argument("--cookies-path", default=str(COOKIES_PATH))
    parser.add_argument("--cookie-header-path", default=str(COOKIE_HEADER_PATH))
    parser.add_argument("--login-info-path", default=str(LOGIN_INFO_PATH))
    parser.add_argument("--login-url", default=LOGIN_URL)
    parser.add_argument(
        "--use-browser-session",
        action="store_true",
        help="Log in with Selenium Chrome and fetch the odds API from that same browser session.",
    )
    parser.add_argument("--output", help="Optional .csv or .parquet output path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = fetch_oddsshopper_live_hr_odds(
        target_date=date.fromisoformat(args.date),
        state=args.state,
        league=args.league,
        market_name=args.market_name,
        sportsbook_filter=args.bookmakers,
        cookies_path=Path(args.cookies_path),
        cookie_header_path=Path(args.cookie_header_path),
        use_browser_session=bool(args.use_browser_session),
        login_info_path=Path(args.login_info_path),
        login_url=args.login_url,
    )

    if df.empty:
        print("No live odds rows found.")
        return

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.output_dir) / f"mlb_hr_live_oddsshopper_{args.date}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

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

from datetime import date

import pandas as pd
import streamlit as st

from live_dashboard_utils import (
    DEFAULT_BOOKMAKERS,
    build_all_loaded_bets_table,
    build_game_roster_map,
    build_live_candidate_rows,
    build_live_matchups,
    build_live_weather_df,
    build_ranked_bets_table,
    fetch_live_hr_odds,
    format_bet_for_social,
    load_model_bundle,
    load_precomputed_live_feature_cache,
    prepare_odds_features,
    score_live_candidates,
)
from pullOddsShopperLive import fetch_oddsshopper_live_hr_odds
from pullLiveOdds import get_api_key_diagnostics


st.set_page_config(page_title="MLB HR Dashboard", layout="wide")

with st.expander("Odds API Diagnostics", expanded=False):
    odds_api_diag = get_api_key_diagnostics()
    st.write("Resolved source:", odds_api_diag["resolved_source"])
    st.write("Secrets found:", odds_api_diag["secret_keys_present"])
    st.write("Env found:", odds_api_diag["env_keys_present"])
    st.write("Local apiKey.txt exists:", odds_api_diag["api_key_file_exists"])

ODDS_BUCKET_BINS = [-10_000, 200, 300, 400, 500, 700, 1000, 2000, 10_000]
ODDS_BUCKET_LABELS = [
    "<= +200",
    "+201 to +300",
    "+301 to +400",
    "+401 to +500",
    "+501 to +700",
    "+701 to +1000",
    "+1001 to +2000",
    "> +2000",
]
PREFERRED_ALLOWED_BUCKETS = {
    "+401 to +500",
    "+701 to +1000",
}


@st.cache_resource
def get_model_bundle():
    return load_model_bundle()


@st.cache_data(show_spinner=False)
def get_precomputed_feature_tables_cached():
    return load_precomputed_live_feature_cache()


@st.cache_data(show_spinner=False, ttl=300)
def pull_pitchers_cached(target_date: date, game_type: str):
    return build_live_matchups(target_date=target_date, game_type=game_type or None)


@st.cache_data(show_spinner=False, ttl=300)
def pull_weather_cached(target_date: date, game_type: str):
    return build_live_weather_df(start_date=target_date.isoformat(), game_type=game_type or None)


@st.cache_data(show_spinner=False, ttl=300)
def pull_odds_cached(
    target_date: date,
    bookmakers: str,
    odds_source: str,
    oddsshopper_state: str,
    oddsshopper_browser_session: bool,
    manual_odds_api_key: str,
):
    parts = []
    if odds_source in {"Odds API", "Both"}:
        odds_api_df = fetch_live_hr_odds(
            bookmakers=bookmakers or None,
            snapshot_label="live",
            target_date=target_date,
            api_key_override=manual_odds_api_key or None,
        )
        if not odds_api_df.empty:
            parts.append(odds_api_df)

    if odds_source in {"OddsShopper", "Both"}:
        oddsshopper_df = fetch_oddsshopper_live_hr_odds(
            target_date=target_date,
            state=oddsshopper_state,
            sportsbook_filter=bookmakers or None,
            use_browser_session=oddsshopper_browser_session,
        )
        if not oddsshopper_df.empty:
            parts.append(oddsshopper_df)

    odds_raw_df = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()
    if odds_raw_df.empty:
        odds_features_df = pd.DataFrame()
    else:
        odds_features_df = prepare_odds_features(odds_raw_df)
    return odds_raw_df, odds_features_df


@st.cache_data(show_spinner=False, ttl=300)
def pull_rosters_cached(matchups_df: pd.DataFrame, target_date: date):
    return build_game_roster_map(matchups_df=matchups_df, target_date=target_date)


def ensure_state_defaults():
    for key, default in [
        ("matchups_df", pd.DataFrame()),
        ("weather_df", pd.DataFrame()),
        ("odds_raw_df", pd.DataFrame()),
        ("odds_features_df", pd.DataFrame()),
        ("roster_map_df", pd.DataFrame()),
        ("candidates_df", pd.DataFrame()),
        ("scored_df", pd.DataFrame()),
        ("feature_cache_metadata", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


ensure_state_defaults()

st.title("MLB HR Live Dashboard")
st.caption(
    "Community Cloud deployment should use The Odds API via `ODDS_API_KEY` in Streamlit secrets. "
    "OddsShopper browser-session login is local-only and is not expected to work reliably on Community Cloud."
)

with st.sidebar:
    st.header("Inputs")
    target_date = st.date_input("Target date", value=date.today())
    game_type = st.text_input(
        "MLB game type",
        value="",
        help="Leave blank for all. Use R for regular season, S for spring.",
    )
    bookmakers = st.text_input("Bookmakers", value=DEFAULT_BOOKMAKERS)
    odds_source = st.selectbox("Odds source", ["Odds API", "OddsShopper", "Both"], index=1)
    oddsshopper_state = st.text_input("OddsShopper state", value="PA")
    oddsshopper_browser_session = st.checkbox(
        "Use browser session for OddsShopper",
        value=True,
        help="Logs into OddsShopper in Chrome and fetches the odds API from that same browser session.",
    )
    manual_odds_api_key = st.text_input(
        "Odds API key override",
        value="",
        type="password",
        help="Temporary fallback if Streamlit secrets are not being picked up. Leave blank to use secrets/env/file.",
    )
    boost_book = st.text_input("Boost bookmaker", value="draftkings")
    boost_pct = st.number_input("Boost percent", min_value=0.0, max_value=200.0, value=0.0, step=5.0)
    early_mode = st.checkbox(
        "Early mode (0.25u)",
        value=target_date.month in {3, 4},
        help="When enabled, copy text uses a flat 0.25 unit sizing.",
    )
    min_edge = st.number_input("Min edge %", min_value=-20.0, max_value=100.0, value=2.0, step=0.5)
    min_books = st.number_input("Min books", min_value=1, max_value=20, value=1, step=1)
    use_preferred_buckets = st.checkbox(
        "Limit to preferred buckets",
        value=True,
        help="Use the preferred live filter from backtesting: +401 to +500 and +701 to +1000.",
    )

    st.header("Actions")
    pull_pitchers_button = st.button("Pull Pitchers", width="stretch")
    pull_weather_button = st.button("Pull Weather", width="stretch")
    pull_odds_button = st.button("Pull Odds", width="stretch")
    run_model_button = st.button("Run Model", type="primary", width="stretch")
    clear_live_cache_button = st.button("Clear Live Cache", width="stretch")
    st.caption(
        "For authenticated OddsShopper pulls, either run "
        "`python3 oddshopper_auth.py` once first to save cookies, or enable the browser-session option."
    )
    if odds_source in {"OddsShopper", "Both"}:
        st.warning(
            "OddsShopper is best treated as a local workflow. For Streamlit Community Cloud, use `Odds API` as the source."
        )

if clear_live_cache_button:
    pull_pitchers_cached.clear()
    pull_weather_cached.clear()
    pull_odds_cached.clear()
    pull_rosters_cached.clear()
    st.success("Cleared cached live pulls.")

if pull_pitchers_button:
    with st.spinner("Pulling probable pitchers..."):
        st.session_state["matchups_df"] = pull_pitchers_cached(target_date, game_type)
        st.session_state["roster_map_df"] = pd.DataFrame()
    st.success(f"Loaded {len(st.session_state['matchups_df']):,} matchup rows.")

if pull_weather_button:
    with st.spinner("Pulling live weather..."):
        st.session_state["weather_df"] = pull_weather_cached(target_date, game_type)
    st.success(f"Loaded {len(st.session_state['weather_df']):,} weather rows.")

if pull_odds_button:
    with st.spinner("Pulling live odds..."):
        odds_raw_df, odds_features_df = pull_odds_cached(
            target_date,
            bookmakers,
            odds_source,
            oddsshopper_state,
            oddsshopper_browser_session,
            manual_odds_api_key,
        )
        st.session_state["odds_raw_df"] = odds_raw_df
        st.session_state["odds_features_df"] = odds_features_df
    if st.session_state["odds_raw_df"].empty:
        st.warning("No live odds rows were returned for that bookmaker/date combination.")
    else:
        st.success(f"Loaded {len(st.session_state['odds_raw_df']):,} raw odds rows.")

if run_model_button:
    with st.spinner("Preparing history and scoring live slate..."):
        try:
            latest_batter_df, latest_pitcher_df, latest_pitcher_split_df, cache_metadata = (
                get_precomputed_feature_tables_cached()
            )
        except FileNotFoundError as exc:
            st.error(
                f"{exc}. Build the cache first with "
                "`python3 build_live_feature_cache.py --date YYYY-MM-DD`."
            )
            st.stop()

        if st.session_state["matchups_df"].empty:
            st.session_state["matchups_df"] = pull_pitchers_cached(target_date, game_type)
        if st.session_state["weather_df"].empty:
            st.session_state["weather_df"] = pull_weather_cached(target_date, game_type)
        if st.session_state["odds_features_df"].empty or st.session_state["odds_raw_df"].empty:
            odds_raw_df, odds_features_df = pull_odds_cached(
                target_date,
                bookmakers,
                odds_source,
                oddsshopper_state,
                oddsshopper_browser_session,
                manual_odds_api_key,
            )
            st.session_state["odds_raw_df"] = odds_raw_df
            st.session_state["odds_features_df"] = odds_features_df

        if st.session_state["roster_map_df"].empty and not st.session_state["matchups_df"].empty:
            st.session_state["roster_map_df"] = pull_rosters_cached(
                st.session_state["matchups_df"],
                target_date,
            )

        candidates_df = build_live_candidate_rows(
            target_date=target_date,
            latest_batter_df=latest_batter_df,
            latest_pitcher_df=latest_pitcher_df,
            latest_pitcher_split_df=latest_pitcher_split_df,
            odds_features_df=st.session_state["odds_features_df"],
            matchups_df=st.session_state["matchups_df"],
            weather_df=st.session_state["weather_df"],
            roster_map_df=st.session_state["roster_map_df"],
        )
        scored_df = score_live_candidates(candidates_df, get_model_bundle())
        st.session_state["candidates_df"] = candidates_df
        st.session_state["scored_df"] = scored_df
        st.session_state["feature_cache_metadata"] = cache_metadata
    st.success(f"Scored {len(st.session_state['scored_df']):,} players.")

status1, status2, status3, status4 = st.columns(4)
status1.metric("Pitchers", f"{len(st.session_state['matchups_df']):,}")
status2.metric("Weather", f"{len(st.session_state['weather_df']):,}")
status3.metric("Odds", f"{len(st.session_state['odds_raw_df']):,}")
status4.metric("Scored", f"{len(st.session_state['scored_df']):,}")

if st.session_state["scored_df"].empty:
    st.info("Use the separate pull buttons, then run the model when you’re ready.")
    with st.expander("Pulled Inputs", expanded=True):
        st.markdown("**Probable Pitchers / Matchups**")
        st.dataframe(st.session_state["matchups_df"], use_container_width=True, hide_index=True)
        st.markdown("**Live Weather**")
        st.dataframe(st.session_state["weather_df"], use_container_width=True, hide_index=True)
        st.markdown("**Live Raw Odds**")
        st.dataframe(st.session_state["odds_raw_df"], use_container_width=True, hide_index=True)
    st.stop()

ranked_df = build_ranked_bets_table(
    scored_df=st.session_state["scored_df"],
    raw_odds_df=st.session_state["odds_raw_df"],
    boost_book=boost_book.strip() or None,
    boost_pct=float(boost_pct),
)

if ranked_df.empty:
    st.warning("No ranked bets were produced.")
    with st.expander("Debug counts", expanded=True):
        st.write({
            "raw_odds_rows": len(st.session_state["odds_raw_df"]),
            "odds_feature_rows": len(st.session_state["odds_features_df"]),
            "matchups_rows": len(st.session_state["matchups_df"]),
            "weather_rows": len(st.session_state["weather_df"]),
            "roster_map_rows": len(st.session_state["roster_map_df"]),
            "candidate_rows": len(st.session_state["candidates_df"]),
            "scored_rows": len(st.session_state["scored_df"]),
        })
    st.stop()

current_et = pd.Timestamp.now(tz="America/New_York")
if "game_time_et" in ranked_df.columns:
    ranked_df = ranked_df[pd.to_datetime(ranked_df["game_time_et"], errors="coerce") > current_et].copy()

ranked_df["odds_bucket"] = pd.cut(
    ranked_df["best_book_odds"],
    bins=ODDS_BUCKET_BINS,
    labels=ODDS_BUCKET_LABELS,
    include_lowest=True,
)

if use_preferred_buckets:
    ranked_df = ranked_df[ranked_df["odds_bucket"].astype(str).isin(PREFERRED_ALLOWED_BUCKETS)].copy()

filtered_df = ranked_df[
    (ranked_df["edge_best_book"] >= min_edge / 100.0) &
    (ranked_df["books_available"] >= min_books)
].copy()

if filtered_df.empty:
    st.warning("No bets passed the current filters.")
    st.dataframe(ranked_df, use_container_width=True, hide_index=True)
    st.stop()

top1, top2, top3, top4 = st.columns(4)
top1.metric("Raw odds rows", f"{len(st.session_state['odds_raw_df']):,}")
top2.metric("Scored players", f"{len(st.session_state['scored_df']):,}")
top3.metric("Ranked bets", f"{len(filtered_df):,}")
top4.metric("Best edge", f"{filtered_df['edge_best_book'].max() * 100:.2f}%")

st.subheader("All Loaded Bets")
all_bets_df = build_all_loaded_bets_table(
    scored_df=st.session_state["scored_df"],
    raw_odds_df=st.session_state["odds_raw_df"],
    boost_book=boost_book.strip() or None,
    boost_pct=float(boost_pct),
)
if "game_time_et" in all_bets_df.columns:
    all_bets_df = all_bets_df[pd.to_datetime(all_bets_df["game_time_et"], errors="coerce") > current_et].copy()

all_bets_df["odds_bucket"] = pd.cut(
    all_bets_df["effective_price"],
    bins=ODDS_BUCKET_BINS,
    labels=ODDS_BUCKET_LABELS,
    include_lowest=True,
)
if use_preferred_buckets:
    all_bets_df = all_bets_df[all_bets_df["odds_bucket"].astype(str).isin(PREFERRED_ALLOWED_BUCKETS)].copy()
all_bets_display_df = all_bets_df.copy()
all_bets_display_df["Pitcher"] = all_bets_display_df.apply(
    lambda row: f"{row['opposing_pitcher_name']}, {row['p_throws']}" if pd.notna(row.get("opposing_pitcher_name")) and pd.notna(row.get("p_throws")) else row.get("opposing_pitcher_name"),
    axis=1,
)
all_bets_display_df = all_bets_display_df.drop(columns=[
    "game_date",
    "game_time_utc",
    "home_team",
    "away_team",
    "opposing_pitcher_name",
    "p_throws",
    "boost_applied",
], errors="ignore")
all_bets_display_df = all_bets_display_df.rename(columns={
    "player_name": "Player",
    "game_time_et": "First pitch ET",
    "batting_team": "Team",
    "bookmaker": "Book",
    "bookmaker_title": "Sportsbook",
    "price": "Raw Odds",
    "effective_price": "Boosted Odds",
    "effective_implied_prob": "Book %",
    "liquidity": "Liquidity",
    "wind_direction": "Wind Dir",
    "edge_vs_book": "Edge",
})
st.dataframe(all_bets_display_df, use_container_width=True, hide_index=True)

st.subheader("Best Bets")
display_df = filtered_df.copy().reset_index(drop=True)
display_df["Pitcher"] = display_df.apply(
    lambda row: f"{row['opposing_pitcher_name']}, {row['p_throws']}" if pd.notna(row.get("opposing_pitcher_name")) and pd.notna(row.get("p_throws")) else row.get("opposing_pitcher_name"),
    axis=1,
)
display_df["best_book_boosted_odds"] = display_df["best_book_odds"]
display_df = display_df.drop(columns=[
    "game_date",
    "game_time_utc",
    "home_team",
    "away_team",
    "best_book",
    "boost_applied",
    "p_throws",
    "opposing_pitcher_name",
    "best_book_odds",
], errors="ignore")
display_df = display_df.rename(columns={
    "player_name": "Player",
    "game_time_et": "First pitch ET",
    "batting_team": "Team",
    "wind_direction": "Wind Dir",
    "best_book_raw_odds": "Raw Odds",
    "best_book_boosted_odds": "Boosted Odds",
    "best_liquidity": "Liquidity",
})
display_df.insert(0, "select", False)

edited_df = st.data_editor(
    display_df,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "select": st.column_config.CheckboxColumn("Select"),
        "First pitch ET": st.column_config.DatetimeColumn("First pitch ET", format="MMM D, YYYY h:mm A"),
        "model_prob": st.column_config.NumberColumn("Model %", format="%.3f"),
        "best_book_implied_prob": st.column_config.NumberColumn("Book %", format="%.3f"),
        "edge_best_book": st.column_config.NumberColumn("Edge", format="%.3f"),
        "expected_profit_per_unit": st.column_config.NumberColumn("EV/unit", format="%.3f"),
        "temperature_f": st.column_config.NumberColumn("Temp F", format="%.1f"),
        "wind_speed_mph": st.column_config.NumberColumn("Wind MPH", format="%.1f"),
    },
    disabled=[c for c in display_df.columns if c != "select"],
)

selected_positions = edited_df.index[edited_df["select"]].tolist()
if len(selected_positions) > 1:
    st.warning("Select one bet to populate the copy box.")

selected_row = filtered_df.iloc[selected_positions[0]] if len(selected_positions) == 1 else filtered_df.iloc[0]

st.subheader("Copy Box")
unit_size = 0.25 if early_mode else 1.0
social_text = format_bet_for_social(selected_row, unit_size=unit_size)
st.text_area("Selected bet text", value=social_text, height=120)

with st.expander("Selected Bet Details", expanded=True):
    st.write(selected_row.to_frame().rename(columns={selected_row.name: "value"}))

with st.expander("Pulled Inputs"):
    st.markdown("**Probable Pitchers / Matchups**")
    st.dataframe(st.session_state["matchups_df"], use_container_width=True, hide_index=True)
    st.markdown("**Live Weather**")
    st.dataframe(st.session_state["weather_df"], use_container_width=True, hide_index=True)
    st.markdown("**Live Raw Odds**")
    st.dataframe(st.session_state["odds_raw_df"], use_container_width=True, hide_index=True)

with st.expander("Run Notes"):
    cache_metadata = st.session_state.get("feature_cache_metadata")
    st.write({
        "target_date": target_date.isoformat(),
        "bookmakers": bookmakers,
        "odds_source": odds_source,
        "oddsshopper_state": oddsshopper_state,
        "game_type": game_type or None,
        "boost_book": boost_book or None,
        "boost_pct": boost_pct,
        "feature_cache_metadata": cache_metadata,
    })
    st.write(
        "Run this on your network with: "
        "`streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501`"
    )

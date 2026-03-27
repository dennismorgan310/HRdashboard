import re
import unicodedata
import pandas as pd

PLAYER_NAME_ALIASES = {
    "josé ramírez": "jose ramirez",
    "michael harris": "michael harris ii",
    "angel martínez": "angel martinez",
    "maikel garcía": "maikel garcia",
    "jazz chisholm": "jazz chisholm jr",
    "ramón laureano": "ramon laureano",
    "salvador pérez": "salvador perez",
    "bobby witt": "bobby witt jr",
    "wenceel pérez": "wenceel perez",
    "fernando tatís": "fernando tatis jr",
    "luis arráez": "luis arraez",
    "josé altuve": "jose altuve",
    "ronald acuña": "ronald acuna jr",
    "julio rodríguez": "julio rodriguez",
    "jesús sánchez": "jesus sanchez",
    "yandy díaz": "yandy diaz",
    "vladimir guerrero": "vladimir guerrero jr",
    "iván herrera": "ivan herrera",
    "teoscar hernández": "teoscar hernandez",
    "mauricio dubón": "mauricio dubon",
    "eugenio suárez": "eugenio suarez",
    "víctor caratini": "victor caratini",
    "javier báez": "javier baez",
    "josé caballero": "jose caballero",
    "victor scott": "victor scott ii",
    "ramón urías": "ramon urias",
    "yoán moncada": "yoan moncada",
    "carlos narváez": "carlos narvaez",
    "jeremy peña": "jeremy pena",
    "nacho alvarez": "nacho alvarez jr",
    "robert hassell": "robert hassell iii",
    "pedro pagés": "pedro pages",
    "andrés giménez": "andres gimenez",
    "miguel andújar": "miguel andujar",
    "andy ibáñez": "andy ibanez",
    "francisco álvarez": "francisco alvarez",
    "vidal bruján": "vidal brujan",
    "luis robert": "luis robert jr",
    "elías díaz": "elias diaz",
    "nasim nuñez": "nasim nunez",
    "yordan álvarez": "yordan alvarez",
    "lourdes gurriel": "lourdes gurriel jr",
    "christian vázquez": "christian vazquez",
    "enrique hernández": "enrique hernandez",
    "luis urías": "luis urias",
    "rafael marchán": "rafael marchan",
    "martín maldonado": "martin maldonado",
    "josé herrera": "jose herrera",
    "luis vázquez": "luis vazquez",
    "gary sánchez": "gary sanchez",
    "j. p. crawford": "jp crawford",
    "j. t. realmuto": "jt realmuto",
    "michael taylor": "michael a taylor",
    "adolis garcía": "adolis garcia"
}

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

def apply_name_alias(name: str) -> str:
    if pd.isna(name):
        return name
    return PLAYER_NAME_ALIASES.get(name, name)

def american_to_prob(odds):
    if pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def normalize_player_name(name: str) -> str:
    if pd.isna(name):
        return name

    name = str(name).strip().lower()

    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[.\-']", " ", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    name = " ".join(name.split())
    return name


def add_scheduled_game_number(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["game_date", "home_team", "away_team", "game_time", "game_id"])

    game_order = (
        df[["game_id", "game_date", "home_team", "away_team", "game_time"]]
        .drop_duplicates()
        .sort_values(["game_date", "home_team", "away_team", "game_time", "game_id"])
    )

    game_order["scheduled_game_no"] = (
        game_order.groupby(["game_date", "home_team", "away_team"])
        .cumcount() + 1
    )

    df = df.merge(
        game_order[["game_id", "scheduled_game_no"]],
        on="game_id",
        how="left",
    )

    return df


def prepare_odds_features(odds_df: pd.DataFrame, snapshot_label: str | None = None) -> pd.DataFrame:
    odds_df = odds_df.copy()

    if snapshot_label is not None:
        odds_df = odds_df[odds_df["snapshot_label"] == snapshot_label].copy()

    odds_df["game_time"] = pd.to_datetime(odds_df["game_time"], utc=True)
    odds_df["game_date"] = (
        odds_df["game_time"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    odds_df["home_team"] = odds_df.apply(
        lambda row: normalize_team_name_to_abbr(
            row["home_team"],
            row["game_date"],
        ),
        axis=1,
    )
    odds_df["away_team"] = odds_df.apply(
        lambda row: normalize_team_name_to_abbr(
            row["away_team"],
            row["game_date"],
        ),
        axis=1,
    )
    odds_df = odds_df.dropna(subset=["home_team", "away_team"]).copy()
    odds_df = add_scheduled_game_number(odds_df)

    odds_df["implied_prob"] = odds_df["price"].apply(american_to_prob)

    odds_df["player_name_norm"] = (
        odds_df["player"]
        .apply(normalize_player_name)
        .apply(apply_name_alias)
    )

    base_agg = (
        odds_df.groupby(
            ["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm"],
            as_index=False,
        )
        .agg(
            best_odds=("price", "max"),
            market_implied_prob=("implied_prob", "min"),
            mean_implied_prob=("implied_prob", "mean"),
            median_implied_prob=("implied_prob", "median"),
            implied_prob_std=("implied_prob", "std"),
            min_implied_prob=("implied_prob", "min"),
            max_implied_prob=("implied_prob", "max"),
            books_available=("bookmaker", "nunique"),
        )
    )
    best_odds = base_agg

    best_odds["implied_prob_std"] = best_odds["implied_prob_std"].fillna(0.0)
    best_odds["implied_prob_range"] = (
        best_odds["max_implied_prob"] - best_odds["min_implied_prob"]
    )
    best_odds["best_vs_mean_gap"] = (
        best_odds["mean_implied_prob"] - best_odds["market_implied_prob"]
    )
    best_odds["best_vs_median_gap"] = (
        best_odds["median_implied_prob"] - best_odds["market_implied_prob"]
    )

    if snapshot_label is not None:
        best_odds["snapshot_label"] = snapshot_label

    return best_odds


def merge_odds_into_features(game_df: pd.DataFrame, odds_features_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.copy()
    odds_features_df = odds_features_df.copy()

    game_df["game_date"] = pd.to_datetime(game_df["game_date"]).dt.tz_localize(None).dt.normalize()
    odds_features_df["game_date"] = pd.to_datetime(odds_features_df["game_date"]).dt.tz_localize(None).dt.normalize()

    if "player_name" not in game_df.columns:
        raise KeyError(
            f"'player_name' not found in game_df. Available columns: {list(game_df.columns)}"
        )

    game_df["player_name_norm"] = (
                                game_df["player_name"]
                                .apply(normalize_player_name)
                                .apply(apply_name_alias)
                            )

    merge_keys = ["game_date", "player_name_norm"]

    if {"home_team", "away_team", "scheduled_game_no"}.issubset(game_df.columns):
        merge_keys = ["game_date", "home_team", "away_team", "scheduled_game_no", "player_name_norm"]

    merged = game_df.merge(
        odds_features_df,
        on=merge_keys,
        how="left",
    )

    return merged


def add_edge_features(game_df: pd.DataFrame, model_prob_col: str = "model_prob") -> pd.DataFrame:
    game_df = game_df.copy()

    game_df["edge_best"] = game_df[model_prob_col] - game_df["market_implied_prob"]
    game_df["edge_mean"] = game_df[model_prob_col] - game_df["mean_implied_prob"]

    return game_df

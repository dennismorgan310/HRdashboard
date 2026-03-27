def american_to_profit(odds):
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)

def simulate_roi(df, edge_threshold=0.05):
    bets = df[
        (df["edge_best"].notna()) &
        (df["edge_best"] >= edge_threshold)
    ].copy()

    if bets.empty:
        return {
            "bets": 0,
            "roi": 0,
            "profit": 0,
            "win_rate": 0,
        }

    bets["profit_per_win"] = bets["best_odds"].apply(american_to_profit)

    bets["profit"] = bets.apply(
        lambda row: row["profit_per_win"] if row["hr_game"] == 1 else -1,
        axis=1
    )

    total_profit = bets["profit"].sum()
    total_bets = len(bets)

    roi = total_profit / total_bets
    win_rate = (bets["hr_game"] == 1).mean()

    return {
        "bets": total_bets,
        "profit": total_profit,
        "roi": roi,
        "win_rate": win_rate,
    }

from difflib import get_close_matches
import pandas as pd


def build_unmatched_name_report(test_df: pd.DataFrame, odds_features_df: pd.DataFrame) -> pd.DataFrame:
    unmatched = test_df[test_df["market_implied_prob"].isna()].copy()

    rows = []

    for game_date, group in unmatched.groupby("game_date"):
        odds_names = odds_features_df.loc[
            odds_features_df["game_date"] == game_date, "player_name_norm"
        ].dropna().unique().tolist()

        for _, row in group.iterrows():
            player_name = row["player_name"]
            player_name_norm = row["player_name_norm"]

            candidates = get_close_matches(player_name_norm, odds_names, n=5, cutoff=0.75)

            rows.append({
                "game_date": game_date,
                "player_name": player_name,
                "player_name_norm": player_name_norm,
                "candidate_1": candidates[0] if len(candidates) > 0 else None,
                "candidate_2": candidates[1] if len(candidates) > 1 else None,
                "candidate_3": candidates[2] if len(candidates) > 2 else None,
                "candidate_4": candidates[3] if len(candidates) > 3 else None,
                "candidate_5": candidates[4] if len(candidates) > 4 else None,
            })

    return pd.DataFrame(rows)
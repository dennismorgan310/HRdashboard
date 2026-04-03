import argparse
from datetime import date

from live_dashboard_utils import build_and_save_live_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and save the precomputed live feature cache for the Streamlit dashboard."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Target date in YYYY-MM-DD format. Historical data is loaded through the prior day.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2023,
        help="First season year to include in the historical Statcast pull when a full refresh is needed.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore any cached Statcast history and rebuild from the requested start year.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_and_save_live_feature_cache(
        target_date=date.fromisoformat(args.date),
        start_year=args.start_year,
        force_full_refresh=bool(args.full_refresh),
    )
    print("Built live feature cache:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

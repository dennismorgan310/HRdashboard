from pathlib import Path
import pandas as pd


def load_hr_odds_parquets(
    folder_path,
    side=None,
    point=None,
    duplicate_subset=None,
    sort_files=True,
    verbose=True,
):
    """
    Load parquet files from a folder, concatenate them, optionally filter,
    and drop duplicates.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing parquet chunk files.
    side : str or None, default None
        Filter rows to this side value, for example "Over".
        If None, no side filter is applied.
    point : float or int or None, default None
        Filter rows to this point value, for example 0.5.
        If None, no point filter is applied.
    duplicate_subset : list[str] or None, default None
        Columns to use when dropping duplicates.
        If None, drops fully duplicated rows.
    sort_files : bool, default True
        Whether to sort parquet files by filename before loading.
    verbose : bool, default True
        Whether to print basic load info.

    Returns
    -------
    pd.DataFrame
        Concatenated, filtered, deduplicated dataframe.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    parquet_files = list(folder.glob("*.parquet"))

    if sort_files:
        parquet_files = sorted(parquet_files)

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {folder}")

    if verbose:
        print(f"Found {len(parquet_files)} parquet files in {folder}")

    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"Rows before filtering: {len(combined):,}")

    if side is not None:
        combined = combined[combined["side"] == side].copy()

    if point is not None:
        combined = combined[combined["point"] == point].copy()

    if verbose:
        print(f"Rows after filtering: {len(combined):,}")

    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=duplicate_subset).reset_index(drop=True)
    after_dedup = len(combined)

    if verbose:
        print(f"Removed duplicates: {before_dedup - after_dedup:,}")
        print(f"Final rows: {after_dedup:,}")

    return combined
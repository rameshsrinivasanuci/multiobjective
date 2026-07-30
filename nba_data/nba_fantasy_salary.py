"""
NBA Fantasy Salary Model
========================
Builds a gradient boosting model that predicts market-rate salaries
from player statistics, then applies it to any target season to generate
fantasy-game salary values calibrated to that year's actual salary cap.

Works for any year range back to ~2000.

Pipeline
--------
1. Scrape per-game + advanced stats from basketball-reference.com
2. Scrape salaries from basketball-reference.com player pages
   (the all_salaries table in each player's HTML comment block)
3. Filter training set to market-rate players only
4. Normalize salary as fraction of that season's cap
5. Train GradientBoostingRegressor on log(salary_pct) ~ stats + position + age
6. Predict for all players in the target season
7. Rescale to dollars using the target season's actual cap

Install
-------
    pip install pandas requests beautifulsoup4 lxml scikit-learn

Usage
-----
    python nba_fantasy_salary.py --train-start 2015 --train-end 2023 --target 2024
    python nba_fantasy_salary.py --train-start 2000 --train-end 2009 --target 2005
    python nba_fantasy_salary.py --train-start 2018 --train-end 2023 --target 2024 \\
        --stats-cache stats.csv --salary-cache salaries.csv --save
"""

import argparse
import os
import time
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Name cleaning
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """
    Fix player name encoding artifacts that arise from basketball-reference
    serving UTF-8 content with incorrect Latin-1 headers, then normalize
    to ASCII-compatible Unicode (NFC form).
    """
    import unicodedata
    if not isinstance(name, str):
        return str(name)
    # Attempt to fix double-encoded UTF-8 (bytes decoded as latin-1 then re-encoded)
    try:
        fixed = name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        fixed = name
    # Normalize to NFC (canonical composed form)
    return unicodedata.normalize("NFC", fixed).strip()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_RATE_DELAY = 4  # seconds between requests

SALARY_CAP = {
    2000: 34_000_000, 2001: 35_500_000, 2002: 42_500_000, 2003: 40_271_000,
    2004: 43_870_000, 2005: 43_870_000, 2006: 49_500_000, 2007: 53_135_000,
    2008: 55_630_000, 2009: 58_680_000, 2010: 57_700_000, 2011: 58_044_000,
    2012: 58_044_000, 2013: 58_679_000, 2014: 58_679_000, 2015: 63_065_000,
    2016: 70_000_000, 2017: 94_143_000, 2018: 99_093_000, 2019: 101_869_000,
    2020: 109_140_000, 2021: 109_140_000, 2022: 112_414_000, 2023: 123_655_000,
    2024: 136_021_000, 2025: 140_588_000, 2026: 154_647_000,
}

VETERAN_MINIMUM = {
    2000:  287_000, 2001:  301_000, 2002:  315_000, 2003:  367_000,
    2004:  385_000, 2005:  398_762, 2006:  412_718, 2007:  427_163,
    2008:  442_114, 2009:  457_588, 2010:  473_604, 2011:  473_604,
    2012:  473_604, 2013:  490_180, 2014:  507_336, 2015:  525_093,
    2016:  543_471, 2017:  553_736, 2018:  562_493, 2019:  582_180,
    2020:  898_310, 2021:  925_258, 2022:  925_258, 2023:  953_079,
    2024: 1_119_563, 2025: 1_157_153, 2026: 1_272_870,
}

_ROOKIE_SCALE_PCT = 0.115
_MARKET_MIN_PCT   = 0.013


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"   # basketball-reference serves UTF-8; force it
    return resp


def _find_table_in_comment(soup: BeautifulSoup, table_id: str):
    """Find a table hidden inside an HTML comment block."""
    # First try live DOM
    table = soup.find("table", {"id": table_id})
    if table:
        return table
    # Then search comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment_str = str(comment)
        if table_id in comment_str:
            inner = BeautifulSoup(comment_str, "lxml")
            table = inner.find("table", {"id": table_id})
            if table:
                return table
    return None


# ---------------------------------------------------------------------------
# Stats scraping  (basketball-reference league pages)
# ---------------------------------------------------------------------------

# Each entry: (url_slug, [table_ids to try in order])
# Basketball Reference changed table ids between eras:
#   newer seasons: per_game_stats / advanced_stats
#   older seasons: per_game       / advanced
_STAT_MAP = {
    "per_game": ("per_game", ["per_game_stats", "per_game"]),
    "advanced": ("advanced", ["advanced_stats",  "advanced"]),
}


def scrape_stats(season: int, stat_type: str = "per_game") -> pd.DataFrame:
    """Scrape one season of player stats from basketball-reference league pages."""
    url_slug, table_ids = _STAT_MAP[stat_type]
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_{url_slug}.html"
    print(f"    [{season}] {stat_type} …", end=" ", flush=True)

    soup = BeautifulSoup(_get(url).text, "lxml")

    table = None
    for tid in table_ids:
        table = _find_table_in_comment(soup, tid)
        if table is not None:
            break
    if table is None:
        raise ValueError(f"No table found (tried {table_ids}) at {url}")

    df = pd.read_html(StringIO(str(table)))[0]

    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"].drop(columns=["Rk"])

    df.columns = [str(c).upper() for c in df.columns]

    # Fix player name encoding artifacts
    if "PLAYER" in df.columns:
        df["PLAYER"] = df["PLAYER"].astype(str).apply(_clean_name)

    # Consolidate traded players: bref uses "TOT" in older seasons, "2TM"/"3TM" in newer
    if "TEAM" in df.columns:
        is_combined = df["TEAM"].isin(["TOT", "2TM", "3TM", "4TM"])
        traded = df.loc[is_combined, "PLAYER"].unique()
        # Keep only the combined row for traded players; keep all rows for non-traded
        df = df[~df["PLAYER"].isin(traded) | is_combined].reset_index(drop=True)
        # If a player has multiple combined rows (rare), keep the one with most games
        df["G_num"] = pd.to_numeric(df["G"], errors="coerce").fillna(0)
        df = (df.sort_values("G_num", ascending=False)
                .drop_duplicates(subset="PLAYER")
                .drop(columns=["G_num"])
                .reset_index(drop=True))

    df["SEASON"] = season
    print(f"{len(df)} players")
    return df


# ---------------------------------------------------------------------------
# Salary scraping  (basketball-reference individual player pages)
# ---------------------------------------------------------------------------

def _season_label_to_end_year(label: str) -> int:
    """Convert '2023-24' → 2024, '2003-04' → 2004."""
    try:
        start, end_short = label.split("-")
        start_yr = int(start)
        # end_short is 2 digits; century rolls over at "00"
        end_yr = start_yr + 1
        return end_yr
    except Exception:
        return None


def _get_player_index() -> pd.DataFrame:
    """
    Fetch the basketball-reference player index — one page per letter.
    Returns a DataFrame with columns: PLAYER, URL
    where URL is the full path to the player's page.
    """
    print("  Building player index from basketball-reference …")
    rows = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"https://www.basketball-reference.com/players/{letter}/"
        try:
            soup = BeautifulSoup(_get(url).text, "lxml")
            table = soup.find("table", {"id": "players"})
            if table is None:
                continue
            for row in table.find_all("tr"):
                th = row.find("th", {"data-stat": "player"})
                if th and th.find("a"):
                    a = th.find("a")
                    rows.append({
                        "PLAYER": _clean_name(a.text),
                        "URL": "https://www.basketball-reference.com" + a["href"],
                    })
            time.sleep(_RATE_DELAY)
        except Exception as e:
            print(f"    Warning: could not fetch index for '{letter}': {e}")
            continue

    df = pd.DataFrame(rows).drop_duplicates(subset="PLAYER")
    print(f"    → {len(df)} players in index")
    return df


def scrape_player_salaries(
    player_names: list,
    seasons: list,
    player_index: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Scrape salary data for a list of players from their individual pages.

    Args:
        player_names:  List of player name strings to fetch.
        seasons:       List of season end years we care about (filters output).
        player_index:  DataFrame with PLAYER and URL columns. If None, fetched automatically.

    Returns:
        DataFrame with columns: PLAYER, SEASON, SALARY (int USD).
    """
    if player_index is None:
        player_index = _get_player_index()

    # Build name → URL lookup
    url_lookup = dict(zip(player_index["PLAYER"], player_index["URL"]))

    target_season_set = set(seasons)
    all_rows = []
    found = 0
    missing = []

    total = len(player_names)
    for i, name in enumerate(player_names, 1):
        url = url_lookup.get(name)
        if url is None:
            missing.append(name)
            continue

        try:
            soup = BeautifulSoup(_get(url).text, "lxml")
            table = _find_table_in_comment(soup, "all_salaries")
            if table is None:
                missing.append(name)
                time.sleep(_RATE_DELAY)
                continue

            df = pd.read_html(StringIO(str(table)))[0]

            # Drop the Career totals row
            df = df[df["Season"].notna()]
            df = df[df["Season"].str.match(r"^\d{4}-\d{2}$", na=False)]

            # Convert season label to end year
            df["SEASON"] = df["Season"].apply(_season_label_to_end_year)
            df = df[df["SEASON"].isin(target_season_set)]

            if df.empty:
                time.sleep(_RATE_DELAY)
                continue

            # Clean salary
            df["SALARY"] = (
                df["Salary"].astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
            )
            df = df[df["SALARY"].str.match(r"^\d+$")]
            df["SALARY"] = df["SALARY"].astype(int)
            df = df[df["SALARY"] > 0]

            # Normalize name to match stats table spelling
            df["PLAYER"] = _clean_name(name)

            all_rows.append(df[["PLAYER", "SEASON", "SALARY"]])
            found += 1

            if i % 50 == 0:
                print(f"    {i}/{total} players scraped …")

            time.sleep(_RATE_DELAY)

        except Exception as e:
            print(f"    Warning: {name} — {e}")
            missing.append(name)
            time.sleep(_RATE_DELAY)
            continue

    print(f"    → Salary data: {found} players found, {len(missing)} not matched")
    if missing:
        print(f"    Not matched (first 10): {missing[:10]}")

    if not all_rows:
        raise ValueError("No salary data scraped.")

    result = pd.concat(all_rows, ignore_index=True)
    # Keep highest salary if a player appears on multiple teams in one season
    result = (
        result.sort_values("SALARY", ascending=False)
        .drop_duplicates(subset=["PLAYER", "SEASON"])
        .reset_index(drop=True)
    )
    return result


# ---------------------------------------------------------------------------
# Scrape all (stats + salaries)
# ---------------------------------------------------------------------------

def scrape_all(
    seasons: list,
    stats_cache: str = None,
    salary_cache: str = None,
) -> tuple:
    """
    Scrape (or load from cache) stats and salaries for a list of seasons.

    Args:
        seasons:      List of season end years e.g. [2018, ..., 2024]
        stats_cache:  CSV path — load if exists, else scrape and save.
        salary_cache: CSV path — load if exists, else scrape and save.

    Returns:
        (stats_df, salaries_df)
    """
    # --- Stats ---
    if stats_cache and os.path.exists(stats_cache):
        print(f"  Loading stats from cache: {stats_cache}")
        all_stats = pd.read_csv(stats_cache, encoding="utf-8")
    else:
        print("  Scraping stats …")
        pg_frames, adv_frames = [], []
        for s in seasons:
            pg_frames.append(scrape_stats(s, "per_game"))
            time.sleep(_RATE_DELAY)
            adv_frames.append(scrape_stats(s, "advanced"))
            time.sleep(_RATE_DELAY)

        pg  = pd.concat(pg_frames,  ignore_index=True)
        adv = pd.concat(adv_frames, ignore_index=True)

        # Both pg and adv should already be deduped to one row per player+season
        # but drop any remaining duplicates before merging to be safe
        pg  = pg.drop_duplicates(subset=["PLAYER", "SEASON"]).reset_index(drop=True)
        adv = adv.drop_duplicates(subset=["PLAYER", "SEASON"]).reset_index(drop=True)

        # Merge per-game and advanced; keep per-game versions of shared columns
        adv_only = ["PLAYER", "SEASON"] + [
            c for c in adv.columns
            if c not in pg.columns and c not in ("PLAYER","SEASON","TEAM","POS","AGE")
        ]
        all_stats = pd.merge(pg, adv[adv_only], on=["PLAYER", "SEASON"], how="left")
        # Final dedup safety net
        all_stats = all_stats.drop_duplicates(subset=["PLAYER", "SEASON"]).reset_index(drop=True)

        if stats_cache:
            all_stats.to_csv(stats_cache, index=False, encoding="utf-8")
            print(f"  Stats saved → {stats_cache}")

    # --- Salaries ---
    if salary_cache and os.path.exists(salary_cache):
        print(f"  Loading salaries from cache: {salary_cache}")
        all_salaries = pd.read_csv(salary_cache, encoding="utf-8")
    else:
        print("  Scraping salaries from player pages …")
        print("  (First building player index — 26 pages) …")
        player_index = _get_player_index()

        unique_players = all_stats["PLAYER"].unique().tolist()
        print(f"  Fetching salary pages for {len(unique_players)} unique players …")
        print(f"  Estimated time: {len(unique_players) * _RATE_DELAY / 60:.0f}–"
              f"{len(unique_players) * (_RATE_DELAY+1) / 60:.0f} minutes\n")

        all_salaries = scrape_player_salaries(unique_players, seasons, player_index)

        if salary_cache:
            all_salaries.to_csv(salary_cache, index=False, encoding="utf-8")
            print(f"  Salaries saved → {salary_cache}")

    return all_stats, all_salaries


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_PERGAME_FEATURES = [
    "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PTS",
]
_ADVANCED_FEATURES = [
    "PER", "TS%", "USG%", "OWS", "DWS", "WS", "WS/48",
    "OBPM", "DBPM", "BPM", "VORP",
]
_POS_MAP = {
    "PG": 0, "SG": 0, "G": 0, "G-F": 0,
    "SF": 1, "PF": 1, "F": 1, "F-G": 1, "F-C": 1,
    "C": 2, "C-F": 2,
}


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = _PERGAME_FEATURES + _ADVANCED_FEATURES + ["AGE"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    feature_cols = [c for c in _PERGAME_FEATURES + _ADVANCED_FEATURES if c in df.columns]
    df[feature_cols] = df[feature_cols].fillna(0)
    if "AGE" in df.columns:
        df["AGE"] = df["AGE"].fillna(df["AGE"].median())
    pos_col = "POS" if "POS" in df.columns else None
    if pos_col:
        df["POS_ENC"] = (
            df[pos_col].astype(str).str.strip().str.upper()
            .map(_POS_MAP).fillna(1)
        )
    else:
        df["POS_ENC"] = 1
    return df


def _get_feature_cols(df: pd.DataFrame) -> list:
    base = _PERGAME_FEATURES + _ADVANCED_FEATURES + ["AGE", "POS_ENC", "SEASON"]
    return [c for c in base if c in df.columns]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(
    stats_df: pd.DataFrame,
    salaries_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple:
    """
    Train a GradientBoostingRegressor to predict log(salary / cap).
    Training set excludes rookie-scale and veteran-minimum contracts.

    Returns:
        (model, feature_cols)
    """
    # Merge stats + salaries
    merged = pd.merge(stats_df, salaries_df, on=["PLAYER", "SEASON"], how="inner")
    merged["CAP"] = merged["SEASON"].map(SALARY_CAP)
    merged = merged.dropna(subset=["CAP"])
    merged["SALARY_PCT"] = merged["SALARY"] / merged["CAP"]

    # Playing-time filter — remove low-minute and low-game players
    merged["MP"]  = pd.to_numeric(merged.get("MP",  0), errors="coerce").fillna(0)
    merged["G"]   = pd.to_numeric(merged.get("G",   0), errors="coerce").fillna(0)
    merged = merged[(merged["MP"] >= 10) & (merged["G"] >= 30)]

    # Market-rate filter
    is_vet_min  = merged["SALARY_PCT"] < _MARKET_MIN_PCT
    is_rookie   = (
        (merged["SALARY_PCT"] < _ROOKIE_SCALE_PCT) &
        (pd.to_numeric(merged.get("AGE", 25), errors="coerce").fillna(25) <= 24)
    )
    train = merged[~is_vet_min & ~is_rookie].copy()

    if verbose:
        print(f"\n  Training set: {len(train)} player-seasons "
              f"({len(merged)-len(train)} excluded as rookie/minimum)")
        print(f"  Seasons: {sorted(train['SEASON'].unique())}")

    train = _prepare_features(train)
    feature_cols = _get_feature_cols(train)

    X = train[feature_cols].values
    y = np.log(train["SALARY_PCT"].clip(lower=1e-4))

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    if verbose:
        cv_r2  = cross_val_score(model, X, y, cv=5, scoring="r2")
        y_pred = model.predict(X)
        print(f"\n  Model performance:")
        print(f"    Training R²:           {r2_score(y, y_pred):.3f}")
        print(f"    Cross-val R² (5-fold): {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
        mae_pct = mean_absolute_error(np.exp(y), np.exp(y_pred))
        print(f"    MAE (dollar equiv @ $120M cap): ${mae_pct * 120_000_000:,.0f}")

        importances = pd.Series(model.feature_importances_, index=feature_cols)
        print(f"\n  Top 10 feature importances:")
        for feat, imp in importances.nlargest(10).items():
            print(f"    {feat:<12} {imp:.3f}")

    return model, feature_cols


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_fantasy_salaries(
    model,
    feature_cols: list,
    stats_df: pd.DataFrame,
    target_season: int,
    salaries_df: pd.DataFrame = None,
) -> tuple:
    """
    Predict fantasy salaries for all players in the target season.

    Returns:
        (result_df, cap)
    """
    cap = SALARY_CAP.get(target_season)
    if cap is None:
        raise ValueError(f"No salary cap data for season {target_season}.")

    target = stats_df[stats_df["SEASON"] == target_season].copy()
    if target.empty:
        raise ValueError(f"No stats found for season {target_season}.")

    # Playing-time filter for target year
    target["MP"] = pd.to_numeric(target.get("MP", 0), errors="coerce").fillna(0)
    target["G"]  = pd.to_numeric(target.get("G",  0), errors="coerce").fillna(0)
    before = len(target)
    target = target[(target["MP"] >= 10) & (target["G"] >= 30)].copy()
    print(f"    Target year filter: {before - len(target)} players removed "
          f"(MP < 10 or G < 30), {len(target)} remain")

    target = _prepare_features(target)
    for col in feature_cols:
        if col not in target.columns:
            target[col] = 0

    X = target[feature_cols].values
    log_sal_pct = model.predict(X)
    sal_pct = np.exp(log_sal_pct)

    target["PREDICTED_SALARY_PCT"] = sal_pct
    # Apply $1M salary floor; all salaries are whole numbers
    target["PREDICTED_SALARY"] = np.maximum(
        (sal_pct * cap).round(0).astype(int),
        1_000_000
    )

    # Attach actual salaries for comparison if available
    if salaries_df is not None:
        sal = (
            salaries_df[salaries_df["SEASON"] == target_season][["PLAYER","SALARY"]]
            .rename(columns={"SALARY": "ACTUAL_SALARY"})
        )
        target = pd.merge(target, sal, on="PLAYER", how="left")
        target["ACTUAL_SALARY_PCT"] = (target["ACTUAL_SALARY"] / cap).round(4)

    # Build output: identity + salary cols first, then all stats
    id_cols     = ["PLAYER", "TEAM", "POS", "AGE", "SEASON"]
    salary_cols = ["PREDICTED_SALARY", "PREDICTED_SALARY_PCT"]
    actual_cols = (["ACTUAL_SALARY", "ACTUAL_SALARY_PCT"]
                   if "ACTUAL_SALARY" in target.columns else [])
    skip = set(id_cols + salary_cols + actual_cols + ["POS_ENC", "CAP", "SALARY_PCT", "AWARDS"])
    stat_cols = [c for c in target.columns if c not in skip]

    out_cols = id_cols + salary_cols + actual_cols + stat_cols
    out_cols = [c for c in out_cols if c in target.columns]

    result = (
        target[out_cols]
        .sort_values("PREDICTED_SALARY", ascending=False)
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------------
    # Formatting
    # ----------------------------------------------------------------

    # Salaries → millions with 1 decimal place (e.g. 41.6)
    for col in ["PREDICTED_SALARY", "ACTUAL_SALARY"]:
        if col in result.columns:
            result[col] = (result[col] / 1_000_000).round(1)

    # Salary pct → 4 decimal places
    for col in ["PREDICTED_SALARY_PCT", "ACTUAL_SALARY_PCT"]:
        if col in result.columns:
            result[col] = result[col].round(4)

    # Columns stored as 0-1 decimals that represent percentages:
    # multiply by 100 and show 1 decimal  (e.g. 0.543 → 54.3)
    pct_cols = ["FG%", "3P%", "2P%", "EFG%", "FT%", "TS%", "3PAR", "FTR"]
    for col in pct_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").mul(100).round(1)

    # WS/48 has range ~-0.10 to 0.30 — keep 2 decimal places
    if "WS/48" in result.columns:
        result["WS/48"] = pd.to_numeric(result["WS/48"], errors="coerce").round(2)

    # All remaining numeric stat columns → 1 decimal place
    skip = set(id_cols + salary_cols + actual_cols
               + ["PREDICTED_SALARY", "ACTUAL_SALARY",
                  "PREDICTED_SALARY_PCT", "ACTUAL_SALARY_PCT",
                  "SEASON", "AGE"]
               + pct_cols + ["WS/48"])
    for col in result.columns:
        if col in skip:
            continue
        coerced = pd.to_numeric(result[col], errors="coerce")
        if coerced.notna().mean() > 0.5:
            result[col] = coerced.round(1)

    return result, cap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NBA fantasy salary model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nba_fantasy_salary.py --train-start 2015 --train-end 2023 --target 2024 \\
      --stats-cache stats.csv --salary-cache salaries.csv --save

  python nba_fantasy_salary.py --train-start 2000 --train-end 2009 --target 2005 \\
      --stats-cache stats_2000s.csv --salary-cache salaries_2000s.csv --save

Note: first run scrapes player pages — allow ~30-60 min for 10 seasons.
      Use --salary-cache to avoid re-scraping on subsequent runs.
        """,
    )
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--train-end",   type=int, required=True)
    parser.add_argument("--target",      type=int, required=True)
    parser.add_argument("--stats-cache",  default=None)
    parser.add_argument("--salary-cache", default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    train_seasons = list(range(args.train_start, args.train_end + 1))
    all_seasons   = sorted(set(train_seasons + [args.target]))

    print(f"\n=== NBA Fantasy Salary Model ===")
    print(f"  Training : {args.train_start}–{args.train_end}")
    print(f"  Target   : {args.target}  (cap: ${SALARY_CAP.get(args.target, '?'):,})")
    print(f"  Seasons  : {all_seasons}\n")

    # 1. Data
    stats_df, salaries_df = scrape_all(
        all_seasons,
        stats_cache=args.stats_cache,
        salary_cache=args.salary_cache,
    )

    # 2. Model — train on training seasons only
    train_stats = stats_df[stats_df["SEASON"].isin(train_seasons)]
    train_sal   = salaries_df[salaries_df["SEASON"].isin(train_seasons)]
    print("\nBuilding model …")
    model, feature_cols = build_model(train_stats, train_sal, verbose=True)

    # 3. Predict for target season
    print(f"\nPredicting fantasy salaries for {args.target} …")
    fantasy_df, cap = predict_fantasy_salaries(
        model, feature_cols, stats_df, args.target, salaries_df
    )

    # 4. Display top 50
    print(f"\n{'='*70}")
    print(f"Fantasy Salary List — {args.target-1}–{str(args.target)[-2:]}  "
          f"| Cap: ${cap:,}  | Roster: 8 players")
    print(f"{'='*70}")
    disp = ["PLAYER","TEAM","POS","PTS","AST","TRB","PREDICTED_SALARY","PREDICTED_SALARY_PCT"]
    if "ACTUAL_SALARY" in fantasy_df.columns:
        disp.append("ACTUAL_SALARY")
    disp = [c for c in disp if c in fantasy_df.columns]
    print(fantasy_df[disp].head(50).to_string(index=False))

    # 5. Save
    if args.save:
        fname = f"nba_fantasy_{args.target}.csv"
        fantasy_df.to_csv(fname, index=False, encoding="utf-8")
        print(f"\nSaved: {fname}  ({len(fantasy_df)} players, cap=${cap:,})")

    return fantasy_df, cap, model


if __name__ == "__main__":
    main()

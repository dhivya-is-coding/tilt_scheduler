import re
import numpy as np
import pandas as pd
import math
from collections import defaultdict, Counter
from datetime import datetime, timedelta

gid = '1741745062'
sheet_id = '1hDSENodpAjdOQudeX0HbAXauWn6KMPiK7lkj50Q-uug'

url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

df = pd.read_csv(url)

def extract_by_weektags_and_final_elos(df: pd.DataFrame):
    """
    Fixed-format cleaner for Tilt sheet:
      - First 4 columns (row 0 headers): Player 1 | Score | Player 2 | Score
      - One special "Initial ELO" marker column: row 0 contains "Initial ELO ->"
      - That same column contains week tags ("Week 1", "Week 2", ...) marking the LAST match row of each week
      - The LAST ROW contains final ELOs for all players (to the right of the marker column)
    Returns:
      matches_df: columns [week, player1, player1_score, player2, player2_score]
      final_elos: columns [player, elo]
    """

    # --- 0) Identify the marker ("Initial ELO") column ---
    header_row = df.iloc[0]
    marker_idx_list = [
        i for i, c in enumerate(df.columns)
        if "initial elo" in str(header_row[c]).strip().lower()
    ]
    if not marker_idx_list:
        raise ValueError("Could not find the 'Initial ELO' marker in row 0.")
    marker_idx = marker_idx_list[0]
    # print(marker_idx)
    marker_col = df.columns[marker_idx]

    # --- 1) Validate the match headers (first 4 columns) & build matches block ---
    match_cols = list(df.columns[:4])
    expected_headers = ["Player 1", "Score", "Player 2", "Score"]
    actual_headers = [str(header_row[c]).strip() for c in match_cols]
    if actual_headers != expected_headers:
        raise ValueError(f"First 4 headers do not match {expected_headers}. Found {actual_headers}.")

    # Data rows start at row index 1
    matches = df.loc[1:, match_cols].copy()
    matches.columns = ["player1", "player1_score", "player2", "player2_score"]

    # Normalize types/strings
    for col in ["player1", "player2"]:
        matches[col] = matches[col].astype(str).str.strip()
        matches[col] = matches[col].replace({"nan": np.nan})

    for col in ["player1_score", "player2_score"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    # --- 2) Build week numbers from the marker column's week tags ---
    # The marker column (same rows) carries strings like "Week 1", "Week 2", ...
    tags = df.loc[1:, marker_col].astype(str).str.strip()
    # We'll iterate row-wise; when a row has a "Week X" tag, that row is the LAST match for that week.
    week_nums = []
    current_week = 1

    week_pattern = re.compile(r"week\s*(\d+)", re.IGNORECASE)

    # We need aligned indices with `matches`
    for idx in matches.index:
        cell = str(tags.get(idx, "")).strip()
        # Assign current week to this row
        week_nums.append(current_week)
        # If this row is tagged as Week N, then it's the last match of that week -> increment for next row
        m = week_pattern.search(cell)
        if m:
            # If the tag has an explicit number, sync to that number (defensive & idempotent)
            explicit = int(m.group(1))
            # If explicit < current_week, keep monotonic; else set to explicit
            # (Normally it matches; this keeps us safe if the sheet reorders.)
            current_week = max(current_week, explicit)
            # After tagging this row as week explicit/current_week, advance to the next week for the next row
            current_week += 1

    matches.insert(0, "week", week_nums)

    # Drop rows that aren't real matches (missing both players)
    matches = matches.dropna(subset=["player1", "player2"], how="all").reset_index(drop=True)

    # --- final ELOs from the last non-empty row of the player columns ---

    # 1) Find the Checksum column by column name
    checksum_idx_list = [i for i, c in enumerate(df.columns) if str(c).strip().lower() == "checksum"]
    if not checksum_idx_list:
        raise ValueError("Could not find 'Checksum' column in DataFrame.")
    checksum_idx = checksum_idx_list[0]

    # 2) Player columns are strictly between 'Initial ELO ->' (marker_idx) and 'Checksum'
    player_cols = list(df.columns[marker_idx + 1 : checksum_idx])

    # 3) Build a sub-DF with only player columns, coerce to numeric, drop fully empty rows
    players_df = df.loc[:, player_cols].copy()

    # Coerce everything to numeric (non-numeric -> NaN)
    for c in players_df.columns:
        players_df[c] = pd.to_numeric(players_df[c], errors="coerce")

    # Drop rows where ALL player columns are NaN (blank rows)
    players_df = players_df.dropna(how="all")
    if players_df.empty:
        raise ValueError("All player rows are empty after dropping blanks; cannot compute final ELOs.")

    # 4) Take the last remaining row as the final ELO snapshot
    last_row = players_df.tail(1).iloc[0]

    # 5) Build the final_elos DataFrame
    final_elos = []
    for col in player_cols:
        val = last_row[col]
        if pd.notna(val):
            # collapse duplicate headers like 'Sam.1' -> 'Sam'
            name = re.sub(r"\.\d+$", "", str(col).strip())
            final_elos.append((name, float(val)))

    final_elos = (
        pd.DataFrame(final_elos, columns=["player", "elo"])
          .groupby("player", as_index=False)["elo"]
          .last()
          .sort_values("elo", ascending=False)
          .reset_index(drop=True)
    )

    return matches, final_elos

# matches, final_elos = extract_by_weektags_and_final_elos(df)
# print(matches)
# print(final_elos)

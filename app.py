import json
import ast
from pathlib import Path
from typing import Dict, Any, List, Tuple

# import importlib
# import sys
# import types

# # Ensure a minimal, well-formed pyarrow module so that pandas' optional
# # pyarrow integration does not crash even if pyarrow is a namespace package
# # without a __version__ attribute. We don't rely on real pyarrow features in
# # this app, and Streamlit is configured to use legacy dataframe serialization.
# try:
#     _pa = importlib.import_module("pyarrow")
# except ImportError:
#     _pa = types.ModuleType("pyarrow")
#     _pa.__version__ = "0.0.0"
#     sys.modules["pyarrow"] = _pa
# else:
#     if not hasattr(_pa, "__version__"):
#         _pa.__version__ = "0.0.0"

import pandas as pd
import streamlit as st

import loader as ld
import scheduler as sch


# Fixed slot structure for now; can be made configurable later.
TIME_SLOTS: List[str] = [
    "6:00", "6:15", "6:30", "6:45",
    "7:00", "7:15", "7:30", "7:45",
    "8:00", "8:15", "8:30", "8:45",
    "9:00", "9:15", "9:30", "9:45",
]
TABLES_PER_SLOT = 3
DEFAULT_CONSTRAINTS_PATH = Path("constraints.txt")


def load_constraints_from_text(text: str) -> Dict[str, Any]:
    """Parse a constraints file contents.

    Supports JSON (preferred) or a Python dict literal.
    """
    text = text.strip()
    if not text:
        return {}

    # Try JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: try Python literal dict
    try:
        data = ast.literal_eval(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse constraints file: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Constraints file must define a JSON/dict object at top level.")
    return data


def load_constraints_from_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return load_constraints_from_text(text)


def run_schedule_with_constraints(
    constraints: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the scheduler and return (phase1_df, phase2_df)."""
    # Load league data from Google Sheet via loader.py
    matches_df, elos_df = ld.extract_by_weektags_and_final_elos(ld.df)
    players = elos_df["player"].tolist()
    elos = dict(zip(elos_df["player"], elos_df["elo"]))

    # Extract constraint fields (with sensible defaults)
    time_prefs: Dict[str, Any] = constraints.get("time_prefs", {})
    table_prefs: Dict[str, str] = constraints.get("table_prefs", {})
    omitted_players = set(constraints.get("omitted_players", []))
    min_matches = int(constraints.get("min_matches_per_player", 2))
    max_matches = int(constraints.get("max_matches_per_player", 3))
    lookback_weeks = int(constraints.get("lookback_weeks", 3))
    global_window = constraints.get("global_time_window", ["6:00", "9:45"])
    if isinstance(global_window, (list, tuple)) and len(global_window) == 2:
        global_time_window = (str(global_window[0]), str(global_window[1]))
    else:
        global_time_window = ("6:00", "9:45")

    # Build availability from per-player time preferences
    player_availability = sch.build_player_availability_from_time_prefs(
        players=players,
        time_slots=TIME_SLOTS,
        time_prefs=time_prefs,
    )

    current_week = int(matches_df["week"].max()) + 1

    phase1_records, phase2_records = sch.generate_weekly_schedule(
        players=players,
        elos=elos,
        past_matches=matches_df,
        week=current_week,
        time_slots=TIME_SLOTS,
        player_availability=player_availability,
        omitted_players=omitted_players,
        min_matches_per_player=min_matches,
        max_matches_per_player=max_matches,
        tables_per_slot=TABLES_PER_SLOT,
        lookback_weeks=lookback_weeks,
        global_time_window=global_time_window,
        table_prefs=table_prefs,
    )

    return pd.DataFrame(phase1_records), pd.DataFrame(phase2_records)


def schedule_to_grid(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a flat schedule DataFrame into a grid (rows=time, cols=tables)."""
    # For each table, create two columns: one for player1 and one for player2.
    column_labels: List[str] = []
    for k in range(TABLES_PER_SLOT):
        table_number = k + 1
        column_labels.append(f"Table {table_number} P1")
        column_labels.append(f"Table {table_number} P2")

    rows: List[Dict[str, Any]] = []
    index: List[str] = []

    for t in TIME_SLOTS:
        index.append(t)
        # Start with empty cells for all player columns at this time.
        row_for_time: Dict[str, Any] = {label: "" for label in column_labels}
        if not schedule_df.empty:
            subset = schedule_df[schedule_df["slot_time"] == t]
            for _, match_row in subset.iterrows():
                table_idx = int(match_row["table"])
                table_number = table_idx + 1
                p1_label = f"Table {table_number} P1"
                p2_label = f"Table {table_number} P2"
                row_for_time[p1_label] = match_row["player1"]
                row_for_time[p2_label] = match_row["player2"]
        rows.append(row_for_time)

    grid = pd.DataFrame(rows, index=index, columns=column_labels)
    grid.index.name = "Time"
    return grid


def _make_style_fn(grid: pd.DataFrame, selected_player: str):
    """Create a column styling function for a schedule grid."""
    highlight_style = "background-color: #70bf67; font-weight: 600;"
    border_style = "border-left: 2px solid #999;"

    def _style_column(col: pd.Series) -> list[str]:
        col_name = col.name
        is_p2_col = isinstance(col_name, str) and col_name.endswith("P2")
        styles: list[str] = []

        for idx in col.index:
            style_parts: List[str] = []

            if is_p2_col:
                style_parts.append(border_style)

            if selected_player != "(none)" and isinstance(col_name, str) and col_name.startswith("Table "):
                try:
                    parts = col_name.split()
                    table_number = int(parts[1])
                    p1_col = f"Table {table_number} P1"
                    p2_col = f"Table {table_number} P2"

                    p1_val = grid.at[idx, p1_col] if p1_col in grid.columns else ""
                    p2_val = grid.at[idx, p2_col] if p2_col in grid.columns else ""

                    if p1_val == selected_player or p2_val == selected_player:
                        style_parts.append(highlight_style)
                except (IndexError, ValueError):
                    pass

            styles.append(" ".join(style_parts))

        return styles

    return _style_column


def _show_verification(schedule_df: pd.DataFrame) -> None:
    """Display schedule verification stats."""
    players_series = pd.concat([schedule_df["player1"], schedule_df["player2"]])
    match_counts = players_series.value_counts().sort_index()
    players_with_less_than_2 = match_counts[match_counts < 2]
    players_with_3 = match_counts[match_counts == 3]

    if players_with_less_than_2.empty:
        st.success("All players in the schedule have at least 2 matches.")
    else:
        st.warning("Some players have fewer than 2 matches.")
        fewer_df = (
            players_with_less_than_2
            .reset_index(name="matches")
            .rename(columns={"index": "player"})
        )
        st.dataframe(fewer_df, width="stretch")

    if not players_with_3.empty:
        players3_df = (
            players_with_3
            .reset_index(name="matches")
            .rename(columns={"index": "player"})
        )
        st.write("Players with 3 matches:")
        st.dataframe(players3_df, width="stretch")

    slot_minutes_map = {
        t: int(t.split(":")[0]) * 60 + int(t.split(":")[1])
        for t in TIME_SLOTS
    }

    gaps = []
    for player, count in match_counts.items():
        subset = schedule_df[
            (schedule_df["player1"] == player)
            | (schedule_df["player2"] == player)
        ].sort_values("slot_index")
        times = subset["slot_time"].tolist()
        for i in range(len(times) - 1):
            start_time = times[i]
            end_time = times[i + 1]
            gap_minutes = (
                slot_minutes_map[end_time] - slot_minutes_map[start_time]
            )
            gaps.append(
                {
                    "player": player,
                    "first_match": start_time,
                    "last_match": end_time,
                    "gap_minutes": gap_minutes,
                }
            )

    if gaps:
        total_gap = sum(g["gap_minutes"] for g in gaps)
        avg_gap = total_gap / len(gaps)
        st.write(
            f"Average time between consecutive matches: {avg_gap:.1f} minutes"
        )

        gaps_sorted = sorted(gaps, key=lambda g: g["gap_minutes"], reverse=True)
        top5 = gaps_sorted[:5]
        st.write("Top 5 largest gaps between consecutive matches (per player):")
        st.dataframe(pd.DataFrame(top5), width="stretch")
    else:
        st.info("Not enough data to compute time between matches.")


def main() -> None:
    st.set_page_config(page_title="Tilt Scheduler", layout="wide")

    st.title("Tilt Scheduler")

    # Sidebar: constraints source (file path or upload)
    st.sidebar.header("Constraints")
    default_text = ""
    if DEFAULT_CONSTRAINTS_PATH.exists():
        default_text = DEFAULT_CONSTRAINTS_PATH.read_text(encoding="utf-8")

    uploaded = st.sidebar.file_uploader("Upload constraints .txt/.json (optional)", type=["txt", "json"])

    if uploaded is not None:
        constraints_text = uploaded.read().decode("utf-8")
    else:
        constraints_text = st.sidebar.text_area(
            "Or edit constraints here (JSON or Python dict)",
            value=default_text,
            height=300,
        )

    run_button = st.sidebar.button("Run scheduler")

    constraints: Dict[str, Any] = {}
    parse_error = None
    if constraints_text.strip():
        try:
            constraints = load_constraints_from_text(constraints_text)
        except ValueError as exc:  # noqa: BLE001
            parse_error = str(exc)

    if parse_error:
        st.error(f"Error parsing constraints: {parse_error}")
        return

    # Initialize session state for persisted schedules
    if "phase1_df" not in st.session_state:
        st.session_state["phase1_df"] = None
    if "phase2_df" not in st.session_state:
        st.session_state["phase2_df"] = None

    # Run solver when button is pressed, and store results in session_state
    if run_button:
        with st.spinner("Running scheduler..."):
            try:
                phase1_df, phase2_df = run_schedule_with_constraints(constraints)
                st.session_state["phase1_df"] = phase1_df
                st.session_state["phase2_df"] = phase2_df
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scheduler error: {exc}")
                st.session_state["phase1_df"] = None
                st.session_state["phase2_df"] = None

    phase1_df = st.session_state.get("phase1_df")
    phase2_df = st.session_state.get("phase2_df")

    has_phase1 = phase1_df is not None and not phase1_df.empty
    has_phase2 = phase2_df is not None and not phase2_df.empty

    if has_phase1 or has_phase2:
        # Build combined player list for highlighting
        all_players: set = set()
        if has_phase1:
            all_players.update(phase1_df["player1"])
            all_players.update(phase1_df["player2"])
        if has_phase2:
            all_players.update(phase2_df["player1"])
            all_players.update(phase2_df["player2"])

        selected_player = st.selectbox(
            "Highlight matches for player",
            options=["(none)"] + sorted(all_players),
            index=0,
        )

        # --- Phase 1 Schedule ---
        if has_phase1:
            st.subheader("Phase 1: ELO-Optimal Pairings")
            grid1 = schedule_to_grid(phase1_df)
            style_fn1 = _make_style_fn(grid1, selected_player)
            st.dataframe(grid1.style.apply(style_fn1, axis=0), width="stretch")

        # --- Phase 2 Schedule ---
        if has_phase2:
            st.subheader("Phase 2: Optimized Placement")
            grid2 = schedule_to_grid(phase2_df)
            style_fn2 = _make_style_fn(grid2, selected_player)
            st.dataframe(grid2.style.apply(style_fn2, axis=0), width="stretch")

            # Verification for the final (Phase 2) schedule
            st.subheader("Schedule verification")
            _show_verification(phase2_df)
    else:
        st.info("Provide constraints and click 'Run scheduler' in the sidebar.")


if __name__ == "__main__":
    main()

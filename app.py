import json
import ast
from pathlib import Path
from typing import Dict, Any, List

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


def run_schedule_with_constraints(constraints: Dict[str, Any]) -> pd.DataFrame:
    """Run the scheduler using the given constraints and return a schedule DataFrame."""
    # Load league data from Google Sheet via loader.py
    matches_df, elos_df = ld.extract_by_weektags_and_final_elos(ld.df)
    players = elos_df["player"].tolist()
    elos = dict(zip(elos_df["player"], elos_df["elo"]))

    # Extract constraint fields (with sensible defaults)
    time_prefs: Dict[str, Any] = constraints.get("time_prefs", {})
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

    schedule_records = sch.generate_weekly_schedule(
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
    )

    return pd.DataFrame(schedule_records)


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

    # Initialize session state for persisted schedule
    if "schedule_df" not in st.session_state:
        st.session_state["schedule_df"] = None

    # Run solver when button is pressed, and store result in session_state
    if run_button:
        with st.spinner("Running scheduler..."):
            try:
                st.session_state["schedule_df"] = run_schedule_with_constraints(constraints)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scheduler error: {exc}")
                st.session_state["schedule_df"] = None

    schedule_df = st.session_state.get("schedule_df")

    if schedule_df is not None and not schedule_df.empty:
        # Player selector for highlighting
        players_in_schedule = sorted(
            set(schedule_df["player1"]).union(schedule_df["player2"])
        )

        selected_player = st.selectbox(
            "Highlight matches for player",
            options=["(none)"] + players_in_schedule,
            index=0,
        )

        st.subheader("Schedule")
        grid = schedule_to_grid(schedule_df)

        highlight_style = "background-color: #70bf67; font-weight: 600;"
        border_style = "border-left: 2px solid #999;"

        def _style_column(col: pd.Series) -> list[str]:
            """Return styles for a single column.

            - P2 columns get a left border.
            - When a player is selected, both players in that match are highlighted.
            """
            col_name = col.name
            is_p2_col = isinstance(col_name, str) and col_name.endswith("P2")
            styles: list[str] = []

            for idx in col.index:
                style_parts: List[str] = []

                # Always add a vertical border between P1 and P2 for each table.
                if is_p2_col:
                    style_parts.append(border_style)

                # Highlight both players in any match involving the selected player.
                if selected_player != "(none)" and isinstance(col_name, str) and col_name.startswith("Table "):
                    try:
                        # Column format: "Table N P1" or "Table N P2"
                        parts = col_name.split()
                        table_number = int(parts[1])
                        p1_col = f"Table {table_number} P1"
                        p2_col = f"Table {table_number} P2"

                        p1_val = grid.at[idx, p1_col] if p1_col in grid.columns else ""
                        p2_val = grid.at[idx, p2_col] if p2_col in grid.columns else ""

                        if p1_val == selected_player or p2_val == selected_player:
                            style_parts.append(highlight_style)
                    except (IndexError, ValueError):
                        # If the column name is not in the expected format, skip highlighting logic.
                        pass

                styles.append(" ".join(style_parts))

            return styles

        grid_to_show = grid.style.apply(_style_column, axis=0)

        st.dataframe(grid_to_show, width="stretch")

        # st.subheader("Raw schedule records")
        # raw_df = schedule_df.sort_values(["slot_index", "table"]).reset_index(drop=True)

        # if selected_player != "(none)" and not raw_df.empty:
        #     def _highlight_row(row: pd.Series) -> list[str]:
        #         is_match = (row["player1"] == selected_player) or (
        #             row["player2"] == selected_player
        #         )
        #         style = "background-color: #70bf67; font-weight: 600;" if is_match else ""
        #         return [style] * len(row)

        #     raw_to_show = raw_df.style.apply(_highlight_row, axis=1)
        # else:
        #     raw_to_show = raw_df

        # st.dataframe(raw_to_show, width="stretch")
    else:
        st.info("Provide constraints and click 'Run scheduler' in the sidebar.")


if __name__ == "__main__":
    main()

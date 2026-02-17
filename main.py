import pandas as pd
import scheduler as sch
import loader as ld

# Example: time slots every 15 minutes from 6:00 to 9:45
time_slots = [
    "6:00", "6:15", "6:30", "6:45",
    "7:00", "7:15", "7:30", "7:45",
    "8:00", "8:15", "8:30", "8:45",
    "9:00", "9:15", "9:30", "9:45",
]

matches_df, elos_df = ld.extract_by_weektags_and_final_elos(ld.df)
players = elos_df["player"].tolist()
elos = dict(zip(elos_df["player"], elos_df["elo"]))

# Per-player time preferences: examples
#   "early" / "late" / "mid"
#   "before 7:30" / "after 8:00"
#   "between 7:00-8:30" or "between 7:00 and 8:30"
#   or dicts like {"before": "7:30"}, {"between": ("7:00", "8:30")}
time_prefs = {
    "Amanda": "early",
    "Brett": "late",
    "Chris O": "between 7:00-8:30",
}

player_availability = sch.build_player_availability_from_time_prefs(
    players=players,
    time_slots=time_slots,
    time_prefs=time_prefs,
)

# Per-player table preferences: "1" = table 1 only, "1,2" = tables 1 or 2
table_prefs = {
    "Amanda": "1",       # only table 1
    "Brett": "1,2",      # tables 1 or 2
}

# Example: omitted players for this week
omitted_players = {"Griff"}  # or set()

current_week = matches_df["week"].max() + 1  # schedule next week, for example

phase1_schedule, phase2_schedule = sch.generate_weekly_schedule(
    players=players,
    elos=elos,
    past_matches=matches_df,
    week=current_week,
    time_slots=time_slots,
    player_availability=player_availability,
    omitted_players=omitted_players,
    min_matches_per_player=2,
    max_matches_per_player=3,
    tables_per_slot=3,
    lookback_weeks=3,
    global_time_window=("6:00", "9:45"),  # or None
    table_prefs=table_prefs,
)

print("=== Phase 1: ELO-Optimal Pairings ===")
print(pd.DataFrame(phase1_schedule))
print()
print("=== Phase 2: Optimized Placement ===")
print(pd.DataFrame(phase2_schedule))

from typing import List, Dict, Set, Optional, Any
import pandas as pd
from ortools.sat.python import cp_model


def _time_to_minutes(t: str) -> int:
    # "HH:MM" -> minutes since midnight
    h, m = map(int, t.split(":"))
    return h * 60 + m


def build_player_availability_from_time_prefs(
    players: List[str],
    time_slots: List[str],
    time_prefs: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[int]]:
    """Derive slot indices per player from coarse time preferences.

    Args:
        players: Iterable of player names.
        time_slots: List of slot start times as "HH:MM" strings.
        time_prefs: Optional mapping player -> preference.

    Supported preference formats (per player):
      - None / missing / "any": all slots
      - "early": roughly first third of slots
      - "late": roughly last third of slots
      - "mid": middle third of slots
      - "before HH:MM": all slots with time <= HH:MM
      - "after HH:MM": all slots with time >= HH:MM
      - "between HH:MM-HH:MM" or "between HH:MM and HH:MM":
            slots in that inclusive window
      - dict forms like {"before": "HH:MM"}, {"after": "HH:MM"},
            {"between": ("HH:MM", "HH:MM")} are also accepted.

    Returns:
        Dict[player, List[slot_index]] suitable for generate_weekly_schedule.
    """

    time_prefs = time_prefs or {}
    n = len(time_slots)
    all_slots = list(range(n))
    slot_minutes = [_time_to_minutes(t) for t in time_slots]

    def _slots_for_pref(pref: Any) -> List[int]:
        if pref is None:
            return all_slots

        # String-based preferences
        if isinstance(pref, str):
            label = pref.strip().lower()
            if not label or label in ("any", "all"):
                return all_slots

            if label in ("none", "unavailable", "off"):
                return []

            if label in ("early", "earlier"):
                # First half of available slots
                end = max(1, n // 2)
                return list(range(end))

            if label in ("late", "later"):
                # Second half of available slots
                start = max(0, n // 2)
                return list(range(start, n))

            if label in ("mid", "middle"):
                start = n // 3
                end = (2 * n) // 3
                return list(range(start, end))

            if label.startswith("before "):
                t_str = label.split("before", 1)[1].strip()
                cutoff = _time_to_minutes(t_str)
                return [i for i, m in enumerate(slot_minutes) if m <= cutoff]

            if label.startswith("after "):
                t_str = label.split("after", 1)[1].strip()
                cutoff = _time_to_minutes(t_str)
                return [i for i, m in enumerate(slot_minutes) if m >= cutoff]

            if label.startswith("between "):
                body = label[len("between ") :].strip()
                normalized = (
                    body.replace(" to ", "-")
                    .replace(" and ", "-")
                    .replace(" ", "")
                )
                parts = [p for p in normalized.split("-") if p]
                if len(parts) == 2:
                    start_m = _time_to_minutes(parts[0])
                    end_m = _time_to_minutes(parts[1])
                    if end_m < start_m:
                        start_m, end_m = end_m, start_m
                    return [
                        i
                        for i, m in enumerate(slot_minutes)
                        if start_m <= m <= end_m
                    ]

            # Fallback: if we can't parse, allow all slots.
            return all_slots

        # Dict-based preferences
        if isinstance(pref, dict):
            if "before" in pref:
                cutoff = _time_to_minutes(str(pref["before"]))
                return [i for i, m in enumerate(slot_minutes) if m <= cutoff]
            if "after" in pref:
                cutoff = _time_to_minutes(str(pref["after"]))
                return [i for i, m in enumerate(slot_minutes) if m >= cutoff]
            if "between" in pref:
                start_t, end_t = pref["between"]
                start_m = _time_to_minutes(str(start_t))
                end_m = _time_to_minutes(str(end_t))
                if end_m < start_m:
                    start_m, end_m = end_m, start_m
                return [
                    i
                    for i, m in enumerate(slot_minutes)
                    if start_m <= m <= end_m
                ]

        # Anything else -> treat as no restriction
        return all_slots

    availability: Dict[str, List[int]] = {}
    for p in players:
        pref = time_prefs.get(p)
        # Explicit "none" means this player is unavailable for the week.
        if isinstance(pref, str) and pref.strip().lower() in ("none", "unavailable", "off"):
            availability[p] = []
            continue

        slots = _slots_for_pref(pref)
        # For other malformed preferences, fall back to all slots so we don't
        # accidentally exclude a player.
        availability[p] = slots or all_slots

    return availability


def generate_weekly_schedule(
    players: List[str],
    elos: Dict[str, float],
    past_matches: pd.DataFrame,
    week: int,
    time_slots: List[str],
    player_availability: Dict[str, List[int]],
    omitted_players: Optional[Set[str]] = None,
    min_matches_per_player: int = 2,
    max_matches_per_player: int = 3,
    tables_per_slot: int = 3,
    lookback_weeks: int = 3,
    global_time_window: Optional[tuple[str, str]] = None,
    elo_weight: int = 1,
) -> List[Dict[str, Any]]:
    """
    Returns a list of matches like:
      {
        "slot_index": int,
        "slot_time": str,
        "table": int,        # 0..tables_per_slot-1
        "player1": str,
        "player2": str,
      }

    Hard constraints:
      - Each non-omitted player has >= min_matches_per_player and <= max_matches_per_player.
      - No player is double-booked in a slot.
      - At most one match per table per slot.
      - No opponent pair that played in last `lookback_weeks` weeks.
      - Only schedule within each player's availability and global_time_window.

    Soft objective:
      - Minimize sum(|elo_i - elo_j|) across all scheduled matches (scaled by elo_weight).
    """
    omitted_players = omitted_players or set()

    # Filter slots by global time window if provided
    if global_time_window is not None:
        start_min = _time_to_minutes(global_time_window[0])
        end_min = _time_to_minutes(global_time_window[1])
        allowed_slot_indices = [
            i
            for i, t in enumerate(time_slots)
            if start_min <= _time_to_minutes(t) <= end_min
        ]
    else:
        allowed_slot_indices = list(range(len(time_slots)))

    # Per-player slots after applying the global time window
    player_allowed_slots: Dict[str, List[int]] = {}
    for p in players:
        base_slots = player_availability.get(p, [])
        player_allowed_slots[p] = [
            s for s in allowed_slot_indices if s in base_slots
        ]

    # Players with no allowed slots this week are effectively unavailable
    unavailable_players: Set[str] = {
        p for p, slots in player_allowed_slots.items() if not slots
    }

    # Quick feasibility sanity check: do we have enough total capacity
    # (slots * tables * 2 player-appearances per match) to satisfy the
    # minimum match requirement for all non-omitted, available players?
    active_players = [
        p
        for p in players
        if p not in omitted_players and p not in unavailable_players
    ]
    max_matches_capacity = len(allowed_slot_indices) * tables_per_slot
    required_player_appearances = len(active_players) * min_matches_per_player
    max_player_appearances = 2 * max_matches_capacity
    if required_player_appearances > max_player_appearances:
        raise RuntimeError(
            "Infeasible before building model: not enough slots/tables "
            f"to give each active available player min_matches_per_player={min_matches_per_player}. "
            f"Active players={len(active_players)}, allowed_slots={len(allowed_slot_indices)}, "
            f"tables_per_slot={tables_per_slot} -> max_matches={max_matches_capacity}, "
            f"max_player_appearances={max_player_appearances}, "
            f"required_player_appearances={required_player_appearances}. "
            "Increase slots/tables, lower min_matches_per_player, or omit some players."
        )

    # Map players to indices
    players = list(players)
    player_index: Dict[str, int] = {p: i for i, p in enumerate(players)}
    num_players = len(players)
    num_slots = len(time_slots)

    # Build ELO dict with defaults (0 if missing)
    elo_by_idx = [float(elos.get(p, 0.0)) for p in players]

    # Compute disallowed pairs based on past X weeks
    recent = past_matches[past_matches["week"] >= week - lookback_weeks]
    disallowed_pairs: Set[tuple[str, str]] = set()
    for _, row in recent.iterrows():
        a = row["player1"]
        b = row["player2"]
        if a in player_index and b in player_index:
            # store in a sorted canonical form
            if a <= b:
                disallowed_pairs.add((a, b))
            else:
                disallowed_pairs.add((b, a))

    # Precompute all potential pairs
    pairs: List[tuple[int, int]] = []
    pair_index_map: Dict[tuple[int, int], int] = {}
    for i in range(num_players):
        for j in range(i + 1, num_players):
            name_i = players[i]
            name_j = players[j]
            if (
                name_i in omitted_players
                or name_j in omitted_players
                or name_i in unavailable_players
                or name_j in unavailable_players
            ):
                continue
            # Skip pairs that are disallowed by history
            key = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
            if key in disallowed_pairs:
                continue
            pair_idx = len(pairs)
            pairs.append((i, j))
            pair_index_map[(i, j)] = pair_idx

    num_pairs = len(pairs)

    model = cp_model.CpModel()

    # Decision variables x[pair, slot, table] ∈ {0,1}
    x = {}
    for m_idx, (i, j) in enumerate(pairs):
        name_i = players[i]
        name_j = players[j]
        for s in allowed_slot_indices:
            # Check both players are available in this slot
            if s not in player_allowed_slots[name_i] or s not in player_allowed_slots[name_j]:
                continue
            for k in range(tables_per_slot):
                var = model.NewBoolVar(f"x_m{m_idx}_s{s}_k{k}")
                x[(m_idx, s, k)] = var

    # Helper to safely get variable (None if not created)
    def get_x(m_idx: int, s: int, k: int):
        return x.get((m_idx, s, k), None)

    # Constraints

    # 1) At most one match per table per slot
    for s in allowed_slot_indices:
        for k in range(tables_per_slot):
            vars_here = [
                get_x(m_idx, s, k)
                for m_idx in range(num_pairs)
                if get_x(m_idx, s, k) is not None
            ]
            if vars_here:
                model.Add(sum(vars_here) <= 1)

    # Optionally: at most tables_per_slot matches per slot (across tables)
    for s in allowed_slot_indices:
        vars_here = [
            get_x(m_idx, s, k)
            for m_idx in range(num_pairs)
            for k in range(tables_per_slot)
            if get_x(m_idx, s, k) is not None
        ]
        if vars_here:
            # You can tighten this to == tables_per_slot if you know it's always feasible.
            model.Add(sum(vars_here) <= tables_per_slot)

    # 2) No double-booking: for each player and slot, at most one match
    for p_idx, name_p in enumerate(players):
        for s in allowed_slot_indices:
            vars_here = []
            for m_idx, (i, j) in enumerate(pairs):
                if p_idx not in (i, j):
                    continue
                for k in range(tables_per_slot):
                    v = get_x(m_idx, s, k)
                    if v is not None:
                        vars_here.append(v)
            if vars_here:
                model.Add(sum(vars_here) <= 1)

    # 3) Each pair of players can be scheduled at most once in the week
    for m_idx, (_i, _j) in enumerate(pairs):
        vars_for_pair = []
        for s in allowed_slot_indices:
            for k in range(tables_per_slot):
                v = get_x(m_idx, s, k)
                if v is not None:
                    vars_for_pair.append(v)
        if vars_for_pair:
            model.Add(sum(vars_for_pair) <= 1)

    # 4) Matches per player (min / max or zero if omitted/unavailable)
    matches_per_player = []
    for p_idx, name_p in enumerate(players):
        vars_for_p = []
        for m_idx, (i, j) in enumerate(pairs):
            if p_idx not in (i, j):
                continue
            for s in allowed_slot_indices:
                for k in range(tables_per_slot):
                    v = get_x(m_idx, s, k)
                    if v is not None:
                        vars_for_p.append(v)
        max_possible = len(allowed_slot_indices)  # safe upper bound
        total_var = model.NewIntVar(0, max_possible, f"matches_{name_p}")
        if vars_for_p:
            model.Add(total_var == sum(vars_for_p))
        else:
            model.Add(total_var == 0)
        matches_per_player.append(total_var)

        if name_p in omitted_players or name_p in unavailable_players:
            # Player is not participating this week
            model.Add(total_var == 0)
        else:
            # Ensure required minimum / maximum matches
            model.Add(total_var >= min_matches_per_player)
            model.Add(total_var <= max_matches_per_player)

    # Indicator variables: y[p_idx, s] = 1 if player p has a match in slot s
    y: Dict[tuple[int, int], cp_model.IntVar] = {}
    for p_idx, name_p in enumerate(players):
        for s in allowed_slot_indices:
            vars_here = []
            for m_idx, (i, j) in enumerate(pairs):
                if p_idx not in (i, j):
                    continue
                for k in range(tables_per_slot):
                    v = get_x(m_idx, s, k)
                    if v is not None:
                        vars_here.append(v)
            if not vars_here:
                continue
            y_var = model.NewBoolVar(f"y_p{p_idx}_s{s}")
            y[(p_idx, s)] = y_var
            # y_var == 1 if any of the underlying match vars is 1
            for v in vars_here:
                model.Add(v <= y_var)
            model.Add(sum(vars_here) <= len(vars_here) * y_var)

    # Objective: combine ELO closeness with
    #  - a penalty for using later time slots (prefer earlier starts), and
    #  - a penalty for large gaps between the same player's matches.
    terms = []

    # Precompute slot start times in minutes for time-related objectives.
    slot_minutes = {s: _time_to_minutes(time_slots[s]) for s in allowed_slot_indices}
    min_slot_minute = min(slot_minutes.values()) if slot_minutes else 0

    # 4a) ELO difference component
    for m_idx, (i, j) in enumerate(pairs):
        elo_diff = int(abs(elo_by_idx[i] - elo_by_idx[j]))
        for s in allowed_slot_indices:
            for k in range(tables_per_slot):
                v = get_x(m_idx, s, k)
                if v is not None and elo_weight * elo_diff > 0:
                    terms.append(elo_weight * elo_diff * v)

    # 4b) Time-of-night component: push matches earlier in the evening by
    # penalizing later slots relative to the earliest slot.
    time_weight = 1  # Increase to more strongly prefer earlier slots.
    if time_weight > 0 and slot_minutes:
        for m_idx, (_i, _j) in enumerate(pairs):
            for s in allowed_slot_indices:
                delta_min = slot_minutes[s] - min_slot_minute
                if delta_min <= 0:
                    continue
                for k in range(tables_per_slot):
                    v = get_x(m_idx, s, k)
                    if v is not None:
                        terms.append(time_weight * delta_min * v)

    # 4c) Gap penalty component: for each player, penalize pairs of
    # slots where they both play, weighted by how far apart the slots are.
    gap_weight = 2  # Increase this to prioritize tighter schedules more strongly.
    if gap_weight > 0 and slot_minutes:
        for p_idx, name_p in enumerate(players):
            if name_p in omitted_players or name_p in unavailable_players:
                continue

            player_slots = [s for s in allowed_slot_indices if (p_idx, s) in y]
            player_slots.sort()
            # For each pair of slots (s1, s2) where this player could play,
            # add a cost proportional to the time gap if they actually
            # end up playing in both slots.
            for i1, s1 in enumerate(player_slots):
                y1 = y[(p_idx, s1)]
                for s2 in player_slots[i1 + 1 :]:
                    y2 = y[(p_idx, s2)]
                    # g = 1  <=>  player plays in both s1 and s2
                    g = model.NewBoolVar(f"gap_p{p_idx}_s{s1}_s{s2}")
                    model.Add(g <= y1)
                    model.Add(g <= y2)
                    model.Add(g >= y1 + y2 - 1)

                    minutes_diff = abs(slot_minutes[s2] - slot_minutes[s1])
                    if minutes_diff > 0:
                        terms.append(gap_weight * minutes_diff * g)

    if terms:
        model.Minimize(sum(terms))
    else:
        # No possible matches (shouldn't usually happen)
        model.Minimize(0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0  # adjust if needed

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No feasible schedule found (status={status}).")

    # Extract schedule
    schedule: List[Dict[str, Any]] = []
    for s in allowed_slot_indices:
        for k in range(tables_per_slot):
            for m_idx, (i, j) in enumerate(pairs):
                v = get_x(m_idx, s, k)
                if v is not None and solver.Value(v) == 1:
                    schedule.append(
                        {
                            "slot_index": s,
                            "slot_time": time_slots[s],
                            "table": k,
                            "player1": players[i],
                            "player2": players[j],
                        }
                    )

    # Optional: sort by time then table for nicer output
    schedule.sort(key=lambda r: (r["slot_index"], r["table"]))
    return schedule
import bisect
from typing import List, Dict, Set, Optional, Any, Tuple
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


# ---------------------------------------------------------------------------
# Phase 1: Find ELO-optimal pairings
# ---------------------------------------------------------------------------

def _solve_phase1_pairings(
    pairs: List[Tuple[int, int]],
    players: List[str],
    allowed_slot_indices: List[int],
    player_allowed_slots: Dict[str, List[int]],
    tables_per_slot: int,
    omitted_players: Set[str],
    unavailable_players: Set[str],
    min_matches_per_player: int,
    max_matches_per_player: int,
    elo_by_idx: List[float],
    elo_weight: int,
    time_slots: List[str],
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """Phase 1: Find optimal pairings using ELO objective only.

    Returns:
        Tuple of (selected_pairs, fallback_schedule).
        selected_pairs: list of (player_idx_i, player_idx_j) that should be scheduled.
        fallback_schedule: full schedule with slot/table assignments (used as fallback
            if Phase 2 fails).
    """
    num_pairs = len(pairs)

    model = cp_model.CpModel()

    # Decision variables x[pair, slot, table] ∈ {0,1}
    x: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
    for m_idx, (i, j) in enumerate(pairs):
        name_i = players[i]
        name_j = players[j]
        for s in allowed_slot_indices:
            # Check both players are available in this slot
            if s not in player_allowed_slots[name_i] or s not in player_allowed_slots[name_j]:
                continue
            for k in range(tables_per_slot):
                x[(m_idx, s, k)] = model.NewBoolVar(f"p1_x_m{m_idx}_s{s}_k{k}")

    def get_x(m_idx: int, s: int, k: int):
        return x.get((m_idx, s, k), None)

    # --- Hard Constraints ---

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

    # At most tables_per_slot matches per slot (across tables)
    for s in allowed_slot_indices:
        vars_here = [
            get_x(m_idx, s, k)
            for m_idx in range(num_pairs)
            for k in range(tables_per_slot)
            if get_x(m_idx, s, k) is not None
        ]
        if vars_here:
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
        max_possible = len(allowed_slot_indices)
        total_var = model.NewIntVar(0, max_possible, f"p1_matches_{name_p}")
        if vars_for_p:
            model.Add(total_var == sum(vars_for_p))
        else:
            model.Add(total_var == 0)

        if name_p in omitted_players or name_p in unavailable_players:
            model.Add(total_var == 0)
        else:
            model.Add(total_var >= min_matches_per_player)
            model.Add(total_var <= max_matches_per_player)

    # --- Objective: ELO only ---
    terms = []
    for m_idx, (i, j) in enumerate(pairs):
        elo_diff = int(abs(elo_by_idx[i] - elo_by_idx[j]))
        if elo_weight * elo_diff <= 0:
            continue
        for s in allowed_slot_indices:
            for k in range(tables_per_slot):
                v = get_x(m_idx, s, k)
                if v is not None:
                    terms.append(elo_weight * elo_diff * v)

    if terms:
        model.Minimize(sum(terms))
    else:
        model.Minimize(0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Phase 1: No feasible schedule found (status={status}).")

    # Extract selected pairs and full schedule (as fallback)
    selected_pairs: List[Tuple[int, int]] = []
    fallback_schedule: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[int, int]] = set()

    for s in allowed_slot_indices:
        for k in range(tables_per_slot):
            for m_idx, (i, j) in enumerate(pairs):
                v = get_x(m_idx, s, k)
                if v is not None and solver.Value(v) == 1:
                    if (i, j) not in seen_pairs:
                        selected_pairs.append((i, j))
                        seen_pairs.add((i, j))
                    fallback_schedule.append(
                        {
                            "slot_index": s,
                            "slot_time": time_slots[s],
                            "table": k,
                            "player1": players[i],
                            "player2": players[j],
                        }
                    )

    fallback_schedule.sort(key=lambda r: (r["slot_index"], r["table"]))
    return selected_pairs, fallback_schedule


# ---------------------------------------------------------------------------
# Phase 2: Greedy compact placement of fixed matches
# ---------------------------------------------------------------------------

def _solve_phase2_placement(
    fixed_matches: List[Tuple[int, int]],
    players: List[str],
    allowed_slot_indices: List[int],
    player_allowed_slots: Dict[str, List[int]],
    tables_per_slot: int,
    omitted_players: Set[str],
    unavailable_players: Set[str],
    time_slots: List[str],
) -> List[Dict[str, Any]]:
    """Phase 2: Greedily assign fixed matches to (slot, table) pairs.

    Uses a match-centric greedy approach: iteratively picks the
    globally best (match, slot) pair according to a compactness score.
    Strongly prefers back-to-back placement (adjacent to a player's
    already-placed match) and breaks ties by urgency (players with
    tight availability windows).

    Returns the schedule as a list of dicts, or raises RuntimeError if
    any matches cannot be placed.
    """
    num_matches = len(fixed_matches)
    if num_matches == 0:
        return []

    sorted_slots = sorted(allowed_slot_indices)

    # Precompute per-player allowed slot sets for fast lookup
    player_slot_set: Dict[str, Set[int]] = {
        p: set(slots) for p, slots in player_allowed_slots.items()
    }

    # Build per-player match lists (indices into fixed_matches)
    player_match_list: Dict[int, List[int]] = {}  # player_idx -> [match_idx, ...]
    for m_idx, (i, j) in enumerate(fixed_matches):
        player_match_list.setdefault(i, []).append(m_idx)
        player_match_list.setdefault(j, []).append(m_idx)

    # For each match, precompute which slots are valid (both players available)
    match_valid_slots: Dict[int, Set[int]] = {}
    for m_idx, (i, j) in enumerate(fixed_matches):
        name_i = players[i]
        name_j = players[j]
        valid = set(
            s for s in sorted_slots
            if s in player_slot_set.get(name_i, set())
            and s in player_slot_set.get(name_j, set())
        )
        match_valid_slots[m_idx] = valid

    # --- Mutable state ---
    unplaced: Set[int] = set(range(num_matches))

    # player_in_slot[s] = set of player indices already booked in slot s
    player_in_slot: Dict[int, Set[int]] = {s: set() for s in sorted_slots}
    # slot_table_count[s] = number of tables used in slot s
    slot_table_count: Dict[int, int] = {s: 0 for s in sorted_slots}
    # player_slots[player_idx] = sorted list of slots where player is placed
    player_slots: Dict[int, List[int]] = {}

    schedule: List[Dict[str, Any]] = []

    # Iteratively place matches one at a time, always picking the globally
    # best (match, slot) combination.
    while unplaced:
        best_score = -float("inf")
        best_match: Optional[int] = None
        best_slot: Optional[int] = None

        for m_idx in unplaced:
            i, j = fixed_matches[m_idx]

            for s in match_valid_slots[m_idx]:
                # Hard constraint: table capacity
                if slot_table_count[s] >= tables_per_slot:
                    continue
                # Hard constraint: no double-booking
                if i in player_in_slot[s] or j in player_in_slot[s]:
                    continue

                # --- Scoring ---
                adjacency = 0
                total_gap = 0
                has_anchor = False
                for p_idx in (i, j):
                    placed = player_slots.get(p_idx)
                    if not placed:
                        continue
                    has_anchor = True
                    min_dist = min(abs(s - ps) for ps in placed)
                    if min_dist == 1:
                        adjacency += 1  # back-to-back
                    total_gap += min_dist

                # Urgency: how constrained is this match?
                open_slots = 0
                for vs in match_valid_slots[m_idx]:
                    if slot_table_count[vs] < tables_per_slot:
                        if i not in player_in_slot[vs] and j not in player_in_slot[vs]:
                            open_slots += 1
                urgency = 1.0 / max(open_slots, 1)

                earliness = -s

                score = (
                    adjacency * 100000.0
                    - total_gap * 1000.0
                    + urgency * 500.0
                    + earliness * 1.0
                )

                if score > best_score:
                    best_score = score
                    best_match = m_idx
                    best_slot = s

        if best_match is None:
            break

        # Place best_match in best_slot
        i_best, j_best = fixed_matches[best_match]
        player_in_slot[best_slot].add(i_best)
        player_in_slot[best_slot].add(j_best)
        slot_table_count[best_slot] += 1
        for p_idx in (i_best, j_best):
            if p_idx not in player_slots:
                player_slots[p_idx] = [best_slot]
            else:
                bisect.insort(player_slots[p_idx], best_slot)

        schedule.append(
            {
                "slot_index": best_slot,
                "slot_time": time_slots[best_slot],
                "table": slot_table_count[best_slot] - 1,
                "player1": players[i_best],
                "player2": players[j_best],
            }
        )
        unplaced.remove(best_match)

    # --- Post-processing: local search to reduce gaps ---
    # Try swapping slot assignments between pairs of matches to reduce
    # the total gap across all players.
    def _total_player_gap(sched: List[Dict[str, Any]]) -> float:
        """Weighted sum of all consecutive gaps (in slot indices) for all players.

        Uses squared gaps so that reducing a large gap matters more
        than reducing a small one.
        """
        p_slots_map: Dict[str, List[int]] = {}
        for rec in sched:
            for key in ("player1", "player2"):
                p_slots_map.setdefault(rec[key], []).append(rec["slot_index"])
        total = 0.0
        for slots_list in p_slots_map.values():
            slots_list.sort()
            for k in range(len(slots_list) - 1):
                gap = slots_list[k + 1] - slots_list[k]
                total += gap * gap  # squared: big gaps penalised more
        return total

    def _is_valid_swap(
        sched: List[Dict[str, Any]], idx_a: int, idx_b: int,
    ) -> bool:
        """Check if swapping slot assignments of sched[idx_a] and sched[idx_b] is valid."""
        rec_a, rec_b = sched[idx_a], sched[idx_b]
        slot_a, slot_b = rec_a["slot_index"], rec_b["slot_index"]
        if slot_a == slot_b:
            return False

        # Check availability
        for key in ("player1", "player2"):
            pa = rec_a[key]
            pb = rec_b[key]
            if slot_b not in player_slot_set.get(pa, set()):
                return False
            if slot_a not in player_slot_set.get(pb, set()):
                return False

        # Build what the schedule looks like at slot_a and slot_b after swap
        players_in_slot_a: List[str] = []
        players_in_slot_b: List[str] = []
        for k, rec in enumerate(sched):
            if k == idx_a or k == idx_b:
                continue
            if rec["slot_index"] == slot_a:
                players_in_slot_a.extend([rec["player1"], rec["player2"]])
            elif rec["slot_index"] == slot_b:
                players_in_slot_b.extend([rec["player1"], rec["player2"]])

        # After swap: rec_a goes to slot_b, rec_b goes to slot_a
        players_in_slot_a.extend([rec_b["player1"], rec_b["player2"]])
        players_in_slot_b.extend([rec_a["player1"], rec_a["player2"]])

        # Check no double-booking
        if len(players_in_slot_a) != len(set(players_in_slot_a)):
            return False
        if len(players_in_slot_b) != len(set(players_in_slot_b)):
            return False

        return True

    # Run local-search passes (swap + relocate)
    for _pass in range(10):
        improved = False
        current_gap = _total_player_gap(schedule)

        # --- Swap: exchange slot assignments of two matches ---
        for idx_a in range(len(schedule)):
            for idx_b in range(idx_a + 1, len(schedule)):
                if not _is_valid_swap(schedule, idx_a, idx_b):
                    continue

                old_slot_a = schedule[idx_a]["slot_index"]
                old_time_a = schedule[idx_a]["slot_time"]
                old_slot_b = schedule[idx_b]["slot_index"]
                old_time_b = schedule[idx_b]["slot_time"]

                schedule[idx_a]["slot_index"] = old_slot_b
                schedule[idx_a]["slot_time"] = old_time_b
                schedule[idx_b]["slot_index"] = old_slot_a
                schedule[idx_b]["slot_time"] = old_time_a

                new_gap = _total_player_gap(schedule)
                if new_gap < current_gap:
                    current_gap = new_gap
                    improved = True
                else:
                    schedule[idx_a]["slot_index"] = old_slot_a
                    schedule[idx_a]["slot_time"] = old_time_a
                    schedule[idx_b]["slot_index"] = old_slot_b
                    schedule[idx_b]["slot_time"] = old_time_b

        # --- Relocate: move a match to a different slot ---
        # Build current slot occupancy
        slot_occ: Dict[int, int] = {s: 0 for s in sorted_slots}
        slot_players: Dict[int, Set[str]] = {s: set() for s in sorted_slots}
        for rec in schedule:
            s = rec["slot_index"]
            slot_occ[s] += 1
            slot_players[s].add(rec["player1"])
            slot_players[s].add(rec["player2"])

        for idx_m in range(len(schedule)):
            rec = schedule[idx_m]
            old_s = rec["slot_index"]
            p1_name, p2_name = rec["player1"], rec["player2"]

            for new_s in sorted_slots:
                if new_s == old_s:
                    continue
                # Capacity check: new slot must have room
                if slot_occ[new_s] >= tables_per_slot:
                    continue
                # Availability check
                if new_s not in player_slot_set.get(p1_name, set()):
                    continue
                if new_s not in player_slot_set.get(p2_name, set()):
                    continue
                # Double-booking: neither player in new slot already
                # (must exclude self from old slot)
                new_slot_ps = set(slot_players[new_s])
                if p1_name in new_slot_ps or p2_name in new_slot_ps:
                    continue

                # Try the move
                old_time = rec["slot_time"]
                rec["slot_index"] = new_s
                rec["slot_time"] = time_slots[new_s]

                new_gap = _total_player_gap(schedule)
                if new_gap < current_gap:
                    # Accept: update occupancy
                    slot_occ[old_s] -= 1
                    slot_players[old_s].discard(p1_name)
                    slot_players[old_s].discard(p2_name)
                    slot_occ[new_s] += 1
                    slot_players[new_s].add(p1_name)
                    slot_players[new_s].add(p2_name)
                    current_gap = new_gap
                    improved = True
                    break  # restart search for this match
                else:
                    rec["slot_index"] = old_s
                    rec["slot_time"] = old_time

        if not improved:
            break

    if unplaced:
        raise RuntimeError(
            f"Phase 2 greedy: {len(unplaced)} match(es) could not be placed."
        )

    # Reassign table numbers within each slot (may have changed after swaps)
    schedule.sort(key=lambda r: (r["slot_index"], r["table"]))
    table_counter: Dict[int, int] = {}
    for rec in schedule:
        s = rec["slot_index"]
        rec["table"] = table_counter.get(s, 0)
        table_counter[s] = table_counter.get(s, 0) + 1

    return schedule


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    elo_weight: int = 100,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Two-phase scheduler: ELO-optimal pairings, then compact placement.

    Phase 1 finds the best pairings (minimizing ELO difference) subject to
    all hard constraints. Phase 2 takes those fixed pairings and assigns them
    to time slots / tables, optimizing for player compactness, schedule
    density (no holes), and earlier start times.

    Returns:
        Tuple of (phase1_schedule, phase2_schedule).
        phase1_schedule: schedule with ELO-optimal pairings and naive placement.
        phase2_schedule: same pairings with optimized slot/table placement.

    Hard constraints (both phases):
      - Each non-omitted player has >= min_matches and <= max_matches.
      - No player is double-booked in a slot.
      - At most one match per table per slot.
      - No opponent pair that played in last `lookback_weeks` weeks.
      - Only schedule within each player's availability and global_time_window.

    Phase 1 soft objective:
      - Minimize sum(|elo_i - elo_j|) across all scheduled matches.

    Phase 2 soft objectives (in priority order):
      - Minimize each player's match span (last - first match time).
      - Minimize gaps between consecutive matches per player.
      - Minimize schedule holes (empty table-slots before later-used ones).
      - Prefer earlier time slots.
    """
    omitted_players = omitted_players or set()

    # ---- Shared Setup ----

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

    # Quick feasibility sanity check
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

    # Build ELO dict with defaults (0 if missing)
    elo_by_idx = [float(elos.get(p, 0.0)) for p in players]

    # Compute disallowed pairs based on past X weeks
    recent = past_matches[past_matches["week"] >= week - lookback_weeks]
    disallowed_pairs: Set[tuple[str, str]] = set()
    for _, row in recent.iterrows():
        a = row["player1"]
        b = row["player2"]
        if a in player_index and b in player_index:
            if a <= b:
                disallowed_pairs.add((a, b))
            else:
                disallowed_pairs.add((b, a))

    # Precompute all potential pairs
    pairs: List[tuple[int, int]] = []
    num_players = len(players)
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
            key = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
            if key in disallowed_pairs:
                continue
            pairs.append((i, j))

    # ---- Phase 1: ELO-optimal pairings ----
    selected_pairs, phase1_schedule = _solve_phase1_pairings(
        pairs=pairs,
        players=players,
        allowed_slot_indices=allowed_slot_indices,
        player_allowed_slots=player_allowed_slots,
        tables_per_slot=tables_per_slot,
        omitted_players=omitted_players,
        unavailable_players=unavailable_players,
        min_matches_per_player=min_matches_per_player,
        max_matches_per_player=max_matches_per_player,
        elo_by_idx=elo_by_idx,
        elo_weight=elo_weight,
        time_slots=time_slots,
    )

    # ---- Phase 2: Optimize placement ----
    try:
        phase2_schedule = _solve_phase2_placement(
            fixed_matches=selected_pairs,
            players=players,
            allowed_slot_indices=allowed_slot_indices,
            player_allowed_slots=player_allowed_slots,
            tables_per_slot=tables_per_slot,
            omitted_players=omitted_players,
            unavailable_players=unavailable_players,
            time_slots=time_slots,
        )
    except RuntimeError:
        # Fallback: use Phase 1's slot assignments if Phase 2 fails
        phase2_schedule = phase1_schedule

    return phase1_schedule, phase2_schedule

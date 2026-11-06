"""Descriptive episode and paired metrics; no inflated map-independence claims."""

import numpy as np


def _action_totals(episodes):
    steps = sum(int(row["length"]) for row in episodes)
    turns = sum(int(row.get("turn_count", round(row["turn_fraction"] * row["length"]))) for row in episodes)
    collisions = sum(int(row["collision_count"]) for row in episodes)
    forwards = steps - turns
    return {"total_steps": steps, "collision_count": collisions, "turn_count": turns,
            "forward_count": forwards, "collision_fraction": collisions / steps,
            "turn_fraction": turns / steps, "forward_fraction": forwards / steps,
            "collision_per_forward_action": collisions / forwards if forwards else None,
            "fraction_denominator": "all effective environment actions; collision_per_forward_action uses forward actions"}


def summarize_trials(trials):
    """Accept TrialRecord objects or their JSON-friendly summary dictionaries."""
    rows = [t.episode_summaries if hasattr(t, "episode_summaries") else t["episode_summaries"] for t in trials]
    if not rows:
        raise ValueError("No trials to summarize")
    if any(len(row) != 2 for row in rows):
        raise ValueError("Metrics require exactly two complete episodes per trial")
    episode = []
    for i in range(2):
        successes = [row[i] for row in rows if row[i]["success"]]
        # The real wrapper has only success and horizon terminations. Engine
        # faults raise and are never folded into this timeout count.
        timeout_count = sum(bool(row[i].get("timeout", not row[i]["success"])) for row in rows)
        episode.append({"mean_return": float(np.mean([row[i]["return"] for row in rows])),
                        "success_rate": float(np.mean([row[i]["success"] for row in rows])),
                        "success_count": len(successes), "trial_count": len(rows),
                        "timeout_count": timeout_count, "timeout_rate": timeout_count / len(rows),
                        "mean_length": float(np.mean([row[i]["length"] for row in rows])),
                        "mean_success_length": float(np.mean([row["length"] for row in successes])) if successes else None,
                        "success_length_sample_count": len(successes),
                        "mean_collisions": float(np.mean([row[i]["collision_count"] for row in rows])),
                        "mean_physical_distance": float(np.mean([row[i]["physical_distance"] for row in rows])),
                        "mean_turn_fraction": float(np.mean([row[i]["turn_fraction"] for row in rows])),
                        **_action_totals([row[i] for row in rows])})
    both = [row for row in rows if row[0]["success"] and row[1]["success"]]
    differences = [row[0]["length"] - row[1]["length"] for row in both]
    counts = {"both_failed": 0, "first_failed_second_succeeded": 0, "first_succeeded_second_failed": 0, "both_succeeded": 0}
    labels = {(False, False): "both_failed", (False, True): "first_failed_second_succeeded",
              (True, False): "first_succeeded_second_failed", (True, True): "both_succeeded"}
    for row in rows:
        counts[labels[(bool(row[0]["success"]), bool(row[1]["success"]))]] += 1
    return {"trial_count": len(rows), "mean_trial_return": float(np.mean([sum(e["return"] for e in row) for row in rows])),
            "trial_return_definition": "undiscounted sum over both complete episodes",
            "action_totals": _action_totals([episode for row in rows for episode in row]),
            "episodes": episode, "outcome_counts": counts, "both_success_count": len(both),
            "paired_success_mean_length_improvement": float(np.mean(differences)) if both else None,
            "paired_success_mean_relative_shortening": float(np.mean([(r[0]["length"] - r[1]["length"]) / r[0]["length"] for r in both])) if both else None,
            "paired_success_strictly_shorter_rate": float(np.mean([d > 0 for d in differences])) if both else None,
            "paired_success_not_longer_rate": float(np.mean([d >= 0 for d in differences])) if both else None,
            "all_trial_strictly_shorter_rate": float(np.mean([r[1]["length"] < r[0]["length"] for r in rows])),
            "all_trial_not_longer_rate": float(np.mean([r[1]["length"] <= r[0]["length"] for r in rows])),
            "uncertainty_note": "Descriptive only; correlated configurations are not independent geometry or training-seed replicates."}

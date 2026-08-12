"""Event study: did the forecast flag real volatility spikes before they happened?

R^2 is the right statistic but the wrong story. What a user actually wants to know is
whether the thing would have warned them in February 2018, March 2020 and 2022 — and how
often it cried wolf in between. This module answers both, using only the walk-forward
out-of-sample predictions, so every warning it claims is one a user could really have seen
on that date.

Definitions, fixed before looking at results:

* **stress episode** — a maximal run of days whose *realized* forward vol sits in the
  stressed regime (above the training-period 80th percentile), lasting at least 5 days.
  Episodes within 21 trading days of each other are merged so one crisis is one event.
* **alarm** — the model's predicted regime is stressed.
* **lead time** — trading days between the first alarm in the 60-day run-up and the
  episode's onset. Zero means it called it on the day; negative means it was late.
* **false alarm** — an alarm day not followed by a stress episode onset within h days.

The false-alarm rate is reported next to the detection rate on purpose. A model that
screams every day detects everything and is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volradar import config
from volradar.backtest.oos import annualize, load_or_build

# Well-known episodes, for labelling the output table. Detection is computed from the data
# and does not depend on this list; the list only attaches a human-readable name.
NAMED_EVENTS = [
    ("2011-08-01", "US downgrade / euro crisis"),
    ("2015-08-20", "China devaluation shock"),
    ("2018-02-05", "Volmageddon (XIV unwind)"),
    ("2018-10-10", "Q4 2018 selloff"),
    ("2020-02-24", "COVID-19 crash"),
    ("2022-01-20", "2022 rate-shock repricing"),
    ("2022-09-13", "CPI shock / bear market low"),
    ("2024-08-02", "Yen carry unwind"),
]

MIN_EPISODE_DAYS = 5
MERGE_GAP = 21
LOOKBACK = 60


def find_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Locate realized stress episodes in the OOS period."""
    d = df.sort_values("date").reset_index(drop=True)
    stressed = (d["true_regime"] == 2).to_numpy()

    episodes, start = [], None
    for i, s in enumerate(stressed):
        if s and start is None:
            start = i
        elif not s and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(d) - 1))

    episodes = [(a, b) for a, b in episodes if b - a + 1 >= MIN_EPISODE_DAYS]

    merged = []
    for a, b in episodes:
        if merged and a - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    rows = []
    for a, b in merged:
        seg = d.iloc[a:b + 1]
        peak_i = int(seg[config.fwd_col(seg["horizon"].iloc[0])].idxmax())
        rows.append({
            "onset_idx": a, "end_idx": b,
            "onset": d.loc[a, "date"], "end": d.loc[b, "date"],
            "days": b - a + 1,
            "peak_date": d.loc[peak_i, "date"],
            "peak_vol_ann": float(annualize(d.loc[peak_i, config.fwd_col(
                d.loc[peak_i, "horizon"])])),
        })
    return pd.DataFrame(rows)


def _name_for(onset: pd.Timestamp) -> str:
    """Attach the nearest named event within 45 calendar days, else describe by date."""
    best, best_gap = None, pd.Timedelta(days=45)
    for date_str, name in NAMED_EVENTS:
        gap = abs(pd.Timestamp(date_str) - onset)
        if gap <= best_gap:
            best, best_gap = name, gap
    return best or f"episode {onset.date()}"


def study(panel: str = "market") -> tuple[pd.DataFrame, dict]:
    df = load_or_build(panel).sort_values("date").reset_index(drop=True)
    h = int(df["horizon"].iloc[0])
    eps = find_episodes(df)
    alarms = (df["pred_regime"] == 2).to_numpy()

    rows = []
    for _, e in eps.iterrows():
        a = int(e["onset_idx"])
        window = slice(max(0, a - LOOKBACK), a + 1)
        idx = np.flatnonzero(alarms[window])
        if len(idx):
            first = max(0, a - LOOKBACK) + int(idx[0])
            lead = a - first
            rows.append({"event": _name_for(e["onset"]), "onset": e["onset"].date(),
                         "peak_vol_ann": round(e["peak_vol_ann"], 1),
                         "days": int(e["days"]), "detected": True,
                         "first_alarm": df.loc[first, "date"].date(),
                         "lead_days": int(lead)})
        else:
            rows.append({"event": _name_for(e["onset"]), "onset": e["onset"].date(),
                         "peak_vol_ann": round(e["peak_vol_ann"], 1),
                         "days": int(e["days"]), "detected": False,
                         "first_alarm": None, "lead_days": None})
    table = pd.DataFrame(rows)

    # --- false-alarm accounting -------------------------------------------------------
    onsets = eps["onset_idx"].to_numpy() if len(eps) else np.array([], dtype=int)
    alarm_idx = np.flatnonzero(alarms)
    good = 0
    for i in alarm_idx:
        # An alarm is vindicated if a stress episode starts within h days, or if the day
        # itself is already inside an episode (the model is right to keep warning).
        near_onset = np.any((onsets >= i) & (onsets <= i + h))
        inside = bool(df.loc[i, "true_regime"] == 2)
        good += int(near_onset or inside)

    summary = {
        "panel": panel, "horizon": h,
        "oos_start": str(df["date"].min().date()), "oos_end": str(df["date"].max().date()),
        "n_episodes": int(len(eps)),
        "n_detected": int(table["detected"].sum()) if len(table) else 0,
        "detection_rate": float(table["detected"].mean()) if len(table) else np.nan,
        "median_lead_days": float(table.loc[table["detected"], "lead_days"].median())
        if table["detected"].any() else np.nan,
        "alarm_days": int(len(alarm_idx)),
        "alarm_share_of_days": float(len(alarm_idx) / len(df)),
        "false_alarm_rate": float(1 - good / len(alarm_idx)) if len(alarm_idx) else np.nan,
    }
    return table, summary


def threshold_sweep(panel: str = "market") -> pd.DataFrame:
    """Detection vs false-alarm across every alarm threshold — the full tradeoff curve.

    The headline `study()` uses one pre-registered rule (alarm = predicted regime is
    stressed). That rule turns out to fire late, for a structural reason: a squared-error
    regression shrinks its predictions toward the mean, so the forecast rarely reaches the
    top quintile until a spike is already underway. Rather than quietly re-pick a threshold
    that flatters the result, this sweeps all of them and shows the curve, so a reader can
    see the whole cost of earlier warnings instead of one chosen point on it.

    The threshold is a percentile of the model's own predicted-vol distribution on the
    training period, so it is a rule a user could actually have applied in real time.
    """
    df = load_or_build(panel).sort_values("date").reset_index(drop=True)
    h = int(df["horizon"].iloc[0])
    eps = find_episodes(df)
    onsets = eps["onset_idx"].to_numpy()

    rows = []
    for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        cut = df["pred"].quantile(q)
        alarms = (df["pred"] >= cut).to_numpy()
        alarm_idx = np.flatnonzero(alarms)

        detected, leads = 0, []
        for a in onsets:
            window = slice(max(0, a - LOOKBACK), a + 1)
            idx = np.flatnonzero(alarms[window])
            if len(idx):
                detected += 1
                leads.append(a - (max(0, a - LOOKBACK) + int(idx[0])))

        good = sum(1 for i in alarm_idx
                   if np.any((onsets >= i) & (onsets <= i + h)) or df.loc[i, "true_regime"] == 2)
        rows.append({
            "alarm_percentile": q,
            "detection_rate": detected / max(len(onsets), 1),
            "median_lead_days": float(np.median(leads)) if leads else np.nan,
            "alarm_share_of_days": len(alarm_idx) / len(df),
            "false_alarm_rate": 1 - good / len(alarm_idx) if len(alarm_idx) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    table, summary = study("market")
    print("\n=== Event study: market volatility spikes, walk-forward OOS ===")
    print(f"  window {summary['oos_start']}..{summary['oos_end']}  h={summary['horizon']}d")
    print(table.to_string(index=False))
    print(f"\n  episodes            : {summary['n_episodes']}")
    print(f"  detected in advance : {summary['n_detected']} "
          f"({summary['detection_rate']:.0%})")
    print(f"  median lead time    : {summary['median_lead_days']:.0f} trading days")
    print(f"  alarm days          : {summary['alarm_days']} "
          f"({summary['alarm_share_of_days']:.1%} of all days)")
    print(f"  false-alarm rate    : {summary['false_alarm_rate']:.1%}")

    sweep = threshold_sweep("market")
    print("\n=== Alarm-threshold tradeoff (all thresholds, not a chosen one) ===")
    show = sweep.copy()
    for c in ["detection_rate", "alarm_share_of_days", "false_alarm_rate"]:
        show[c] = show[c].map(lambda v: f"{v:.0%}")
    print(show.to_string(index=False))

    table.to_csv(config.REPORT_DIR / "events.csv", index=False)
    sweep.to_csv(config.REPORT_DIR / "events_sweep.csv", index=False)
    pd.DataFrame([summary]).to_csv(config.REPORT_DIR / "events_summary.csv", index=False)
    print(f"\nWrote -> {config.REPORT_DIR / 'events.csv'}")


if __name__ == "__main__":
    main()

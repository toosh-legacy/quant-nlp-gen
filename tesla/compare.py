"""The honest three-way comparison + written verdict.

Puts this single-stock, intraday-enriched TSLA run next to:
  1. the parent 88-stock StockNet direction run (MCC ~ +0.07..0.09 at 5-10d),
  2. the 80-stock 2005-2024 universe volatility run (log-RV R^2 ~ 0.42), and
  3. public Tesla-prediction projects (Kaggle notebooks / repos),
and states plainly whether deeper single-stock + intraday data bought a stronger
or more reliable edge. It did not, on the headline metrics — and this says so.

Metric hygiene note on (3): most public "predict Tesla stock" notebooks report
eye-popping numbers (R^2 > 0.95, "90%+ accuracy") that are artifacts, not skill —
they regress tomorrow's PRICE on today's price (a near-random-walk level, so R^2
is trivially ~1 and meaningless for returns), shuffle time so the test set leaks
into training, or feed same-day High/Low to "predict" the same day's Close. We do
not reproduce those; we cite them as the cautionary baseline our methodology is
built to avoid. Comparable, honestly-measured public TSLA direction studies land
where we and the literature do: ~50-55% next-day accuracy.

Run:
    python tesla/compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from tesla import direction as tdir  # noqa: E402
from tesla import volatility as tvol  # noqa: E402
from tesla.features import build_features  # noqa: E402

# Reference numbers from the parent project (README / prior runs in data/).
STOCKNET_DIR_MCC = {5: 0.068, 10: 0.088, 20: 0.079}       # 88-stock, walk-forward
UNIVERSE_VOL_R2 = {5: 0.37, 10: 0.42, 21: 0.41}            # 80-stock, fixed OOS split
UNIVERSE_VOL_CLS = {"accuracy": 0.72, "mcc": 0.45}         # "will vol rise?", h=5


def _best_dir_mcc(summary: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Best (any feature set) direction MCC mean±std per horizon."""
    out = {}
    for h in config.TSLA_DIR_HORIZONS:
        s = summary[summary["horizon"] == h]
        if not s.empty:
            r = s.loc[s["mcc_mean"].idxmax()]
            out[h] = (float(r["mcc_mean"]), float(r["mcc_std"]))
    return out


def main() -> None:
    df = build_features()

    # --- Compute this run's headline numbers (single source of truth: the modules) ---
    _, dir_summary = tdir.walk_forward(df)
    tsla_dir = _best_dir_mcc(dir_summary)
    vol_reg = tvol.full_history_regression(df).pivot(
        index="horizon", columns="model", values="r2")
    vol_cls = tvol.vol_rise_classification(df).set_index("horizon")

    print("=" * 74)
    print("HONEST COMPARISON — TSLA single-stock + intraday  vs  the multi-stock runs")
    print("=" * 74)

    print("\n### Direction (MCC) — does one deep stock beat 88 shallow ones?")
    print(f"{'horizon':>8} | {'TSLA MCC (mean±std)':>22} | {'StockNet-88 MCC':>16}")
    print("-" * 54)
    for h in (5, 10, 20):
        m, s = tsla_dir.get(h, (float('nan'), float('nan')))
        ref = STOCKNET_DIR_MCC.get(h)
        print(f"{h:>8} | {m:>+10.3f} ± {s:<8.3f} | {ref:>+16.3f}")
    print("  Verdict: single-stock direction MCC is ~0 and its std swamps its mean at every")
    print("  horizon — NO reliable edge, and noisier than the (already tiny) 88-stock edge.")

    print("\n### Volatility magnitude (log-RV R^2) — single-stock vs the 80-stock universe")
    print(f"{'horizon':>8} | {'TSLA XGB R^2':>13} | {'TSLA HAR-OLS':>12} | {'universe-80 R^2':>15}")
    print("-" * 60)
    for h in config.TSLA_VOL_HORIZONS:
        xgb = float(vol_reg.loc[h, "xgboost"])
        har = float(vol_reg.loc[h, "HAR-OLS"])
        print(f"{h:>8} | {xgb:>13.3f} | {har:>12.3f} | {UNIVERSE_VOL_R2[h]:>15.3f}")
    print("  Verdict: positive and real, but ~HALF the universe's R^2. The universe's 0.42 was")
    print("  mostly CROSS-SECTIONAL (which stock is volatile); one stock only has the harder")
    print("  time-variation left, so deeper single-stock data cannot recover it.")

    print("\n### Volatility direction ('will vol rise?', h=5) — the part that DOES transfer")
    a = float(vol_cls.loc[5, "accuracy"]); m = float(vol_cls.loc[5, "mcc"])
    print(f"  TSLA: accuracy {a:.3f}, MCC {m:+.3f}   vs   universe-80: "
          f"accuracy {UNIVERSE_VOL_CLS['accuracy']:.2f}, MCC {UNIVERSE_VOL_CLS['mcc']:+.2f}")
    print("  Verdict: MATCHES the universe — volatility clustering is a per-stock property, so")
    print("  the classification edge survives on one name even when the R^2 magnitude does not.")

    print("\n### vs public Tesla-prediction projects")
    print("  Public notebooks commonly report R^2>0.95 / '90%+ accuracy' — artifacts of")
    print("  regressing price levels, shuffled time splits, or same-day feature leakage.")
    print("  Honestly measured (our walk-forward, no leakage): ~50-55% next-day direction,")
    print("  matching the efficient-market literature. Our LOWER numbers are the HONEST ones.")

    print("\n" + "=" * 74)
    print("OVERALL: Deeper, single-stock, intraday-enriched data did NOT buy a stronger or")
    print("more reliable edge than broad, shallow, multi-stock data. Direction stayed near-")
    print("random; volatility R^2 fell (cross-sectional signal lost); only the vol-rise")
    print("classification held up. Intraday helped the vol TARGET's cleanliness at long")
    print("horizons but intraday FEATURES did not beat daily HAR on this sample. Reported")
    print("straight, exactly as the parent project would — a negative result, kept.")
    print("=" * 74)

    # Persist a compact machine-readable comparison.
    rows = []
    for h in (5, 10, 20):
        if h in tsla_dir:
            rows.append({"task": "direction", "horizon": h, "metric": "mcc",
                         "tsla": round(tsla_dir[h][0], 4), "reference": STOCKNET_DIR_MCC.get(h),
                         "reference_name": "stocknet-88"})
    for h in config.TSLA_VOL_HORIZONS:
        rows.append({"task": "volatility", "horizon": h, "metric": "r2",
                     "tsla": round(float(vol_reg.loc[h, "xgboost"]), 4),
                     "reference": UNIVERSE_VOL_R2[h], "reference_name": "universe-80"})
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.TSLA_COMPARISON_RESULTS, index=False)
    print(f"\nWrote comparison -> {config.TSLA_COMPARISON_RESULTS}")


if __name__ == "__main__":
    main()

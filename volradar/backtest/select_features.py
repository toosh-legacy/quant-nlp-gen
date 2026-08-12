"""Choose which factor families to ship — decided on pre-2018 folds only.

The ablation showed that adding every family makes the model *worse*, so a choice has to
be made. Making it on the full walk-forward would be selecting on the same window the
README reports, which is the subtler cousin of tuning on the test set.

So selection runs on walk-forward folds strictly before `config.TEST_START`, and the
chosen set is then reported on 2018-2024, which the selection never saw. The scoring
metric is skill versus persistence rather than R^2, because each fold's R^2 denominator is
that year's own variance and is not comparable across years.
"""

from __future__ import annotations


import pandas as pd

from volradar import config
from volradar.backtest import metrics as M
from volradar.features.assemble import feature_families, load_panel
from volradar.models import dataset as D
from volradar.models import estimators as E

SELECTION_YEARS = [y for y in config.WALK_YEARS if y < int(config.TEST_START[:4])]

def candidates(panel: str) -> list[tuple[str, ...]]:
    """A focused shortlist rather than all 2^k subsets.

    The exhaustive sweep is both far too slow on the sector panel and statistically
    self-defeating: with only six selection folds, picking the best of 32 combinations
    mostly selects noise. The ablation already established the ordering — VIX helps, macro
    hurts, the rest are marginal — so this tests a nested sequence from smallest to largest
    plus the two specific hypotheses worth checking, and ties break toward fewer families.
    """
    seq: list[tuple[str, ...]] = [
        (),                                   # core only — the bar to clear
        ("vix",),                             # the family the ablation liked
        ("vix", "breadth"),                   # + cross-sectional correlation
        ("vix", "breadth", "sector_xs"),      # + sector dispersion
        ("vix", "breadth", "sector_xs", "macro"),   # everything, to confirm macro hurts
        ("breadth",),                         # is breadth useful without VIX?
    ]
    if panel == "sector":
        seq += [("vix", "sector_rel"), ("vix", "breadth", "sector_rel")]
    return seq


def score_set(df: pd.DataFrame, features: list[str], h: int, years: list[int]) -> dict:
    sub = D.labelled(df, features, h)
    fit = D.embargoed_train(sub, h, pd.Timestamp(config.VAL_START))
    val = D.time_slice(sub, config.VAL_START, config.VAL_END)
    params = (E.tune_gbm(fit, val, features, h, quiet=True)
              if len(fit) > 200 and len(val) > 100 else E.GBM_DEFAULT)

    skills = []
    for year in years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = D.embargoed_train(sub, h, cutoff)
        test = D.time_slice(sub, cutoff, f"{year + 1}-01-01")
        if len(train) < 400 or len(test) < 60:
            continue
        pred = E.fit_gbm(train, features, h, params).predict(test)
        m = M.regression_metrics(test[config.fwd_col(h)], pred, test[config.persist_col(h)])
        skills.append(m["skill_vs_persistence"])
    s = pd.Series(skills)
    return {"n_folds": len(s), "skill_mean": s.mean(), "skill_std": s.std(ddof=0),
            "skill_min": s.min(), "years_positive": int((s > 0).sum())}


def select(panel: str, h: int | None = None) -> pd.DataFrame:
    h = h or config.HEADLINE_HORIZON
    df = load_panel(panel)
    fams = feature_families(panel)
    rows = []

    print(f"\n=== Family selection: {panel}, h={h}, folds "
          f"{SELECTION_YEARS[0]}-{SELECTION_YEARS[-1]} (pre-test only) ===")
    for combo in candidates(panel):
        feats = list(fams["core"]) + [c for k in combo for c in fams[k]]
        s = score_set(df, feats, h, SELECTION_YEARS)
        rows.append({"panel": panel, "families": "core+" + ("+".join(combo) or "only"),
                     "combo": list(combo), "n_features": len(feats), **s})

    out = pd.DataFrame(rows).sort_values("skill_mean", ascending=False).reset_index(drop=True)
    for _, r in out.head(8).iterrows():
        print(f"  {r['families']:<42} skill {r['skill_mean']:+.3f} "
              f"+/- {r['skill_std']:.3f}  ({r['years_positive']}/{r['n_folds']} yrs +)")
    print(f"  ... {len(out) - 8} more combinations tested")
    return out


def choose(out: pd.DataFrame) -> pd.Series:
    """Pick the SIMPLEST set within one standard error of the best — the 1-SE rule.

    Taking the top mean would be a mistake here. The candidates are separated by ~0.001 to
    0.03 in mean skill while each carries a fold-to-fold standard deviation of 0.05 to 0.22
    over six folds; on those numbers "best" is a coin flip, and the winner usually has the
    *higher* variance. The one-standard-error rule (Breiman et al.) is the standard remedy:
    treat everything within one standard error of the top as tied, then take the smallest.

    Fewer families is not merely tidier — every extra family is another external series
    that can be revised, delisted, or silently truncated by the data provider.
    """
    best = out.iloc[0]
    se = best["skill_std"] / max(best["n_folds"], 1) ** 0.5
    tied = out[out["skill_mean"] >= best["skill_mean"] - se]
    return tied.sort_values(["n_features", "skill_std"]).iloc[0]


def main() -> None:
    config.ensure_dirs()
    all_rows, chosen = [], {}
    for panel in ("market", "sector"):
        out = select(panel)
        all_rows.append(out)
        best, pick = out.iloc[0], choose(out)
        chosen[panel] = pick["combo"]
        if pick["families"] != best["families"]:
            print(f"  best mean was {best['families']} ({best['skill_mean']:+.3f} "
                  f"+/- {best['skill_std']:.3f}); within one standard error, so taking the "
                  f"simpler {pick['families']} ({pick['skill_mean']:+.3f} "
                  f"+/- {pick['skill_std']:.3f})")
        print(f"  -> chosen for {panel}: core + {pick['combo'] or '(nothing)'}")

    pd.concat(all_rows, ignore_index=True).to_csv(
        config.REPORT_DIR / "family_selection.csv", index=False)
    print("\nSet these in config.SELECTED_FAMILIES:")
    for k, v in chosen.items():
        print(f'    "{k}": {v},')


if __name__ == "__main__":
    main()
